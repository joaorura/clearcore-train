"""Pool residente de clipes de ORIGEM na VRAM (`--gpu-pool-gb`): o lote deixa de levar áudio pelo PCIe.

Motivação: na rig (RTX 3090 em PCIe x1, ~0,82 GB/s H2D) o lote B48 do `--gpu-mix` leva ~128 MB de trilhas secas e
ruído por passo (~0,16 s de cópia). Aqui cada GPU guarda um subconjunto do manifest já decodificado e reamostrado a
48 kHz, e a receita (sorteios na CPU, como antes) passa a referenciar `(slot físico, geração, início, comprimento)`;
a GPU recorta, converte e segue o `GpuMixer`.

Formato no pool (`codec`):
  * "exact" (padrão): int16 quando a conversão é sem perda (x·32768 inteiro, o caso de FLAC/WAV PCM 16 bits já a
    48 kHz), senão float32 (MP3 decodificado, fontes reamostradas pelo soxr). O áudio da GPU é bit a bit o do CPU.
  * "int16": tudo em int16; o que não é sem perda leva escala por clipe (pico/32767), erro ≤ meio passo da escala
    (≤ 1,5e-5 do pico do clipe). ~1,8x mais horas no mesmo orçamento, mas não é equivalente amostra a amostra: o
    ganho de nível do mix_v4 chega a ~200x e um quadro que cruza o limiar de atividade muda o nível ativo (medido:
    até 0,37 do pico em 96 misturas; rótulos 0,002% dos quadros). Fica como opção estatística, não como padrão.

Amostragem sem viés (ver `PoolIndex`): os grupos são (falante, faixa de banda) e o ruído. Cada grupo `g` com `N_g`
clipes no manifest mantém `k_g` (1 ≤ k_g ≤ N_g) clipes no pool, um subconjunto uniforme; a renovação troca um membro
sorteado por um não membro sorteado do MESMO grupo (cadeia cuja distribuição estacionária é a uniforme sobre os
subconjuntos de tamanho k_g, e que parte dela). O sorteio de falante/faixa da receita é o mesmo de antes; dentro do
grupo, o clipe é uniforme entre os membros, logo marginalmente uniforme no grupo, igual ao caminho original. Grupo
completo no pool (k_g = N_g): a posição é o próprio sorteio do CPU, e a receita fica idêntica à original.

Memória: arena única de bytes em páginas de `PAGE_BYTES` (int16 = 24000 amostras/página, f32 = 12000), tabela de
páginas por slot físico (clipes de tamanhos diferentes trocam sem fragmentar). Renovação: thread em segundo plano
decodifica o clipe novo, grava em páginas livres numa stream própria, sincroniza e só então publica a entrada na
tabela compartilhada (dupla entrada por posição + bandeira: o worker nunca lê entrada meio escrita). O slot antigo
fica em quarentena até `delay` lotes depois (toda receita que o referenciava já foi misturada) e o evento da última
mistura ter terminado; a geração do slot é conferida em toda mistura (divergência = recarga do disco, contada).
"""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch
from torch import Tensor

from cctrain.data.features import SR
from cctrain.data.gpu.recipe import StageV4RecipeDataset
from cctrain.data.mix import crop_or_pad, loop_to_length
from cctrain.data.mix_v4 import HOP

PAGE_BYTES = 48000
WIDTH = 256  # páginas por slot: até 256·12000 amostras f32 = 64 s a 48 kHz
S16, S32 = PAGE_BYTES // 2, PAGE_BYTES // 4


def encode(x: np.ndarray, codec: str) -> tuple[bool, float, np.ndarray]:
    """`(é_int16, escala, dados)`. Sem perda quando x·32768 é inteiro em [−32768, 32767]."""
    x = np.asarray(x, np.float32)
    q = x.astype(np.float64) * 32768.0
    if x.size and np.all(q == np.rint(q)) and q.min() >= -32768 and q.max() <= 32767:
        return True, 1.0 / 32768.0, q.astype(np.int16)
    if codec == "int16":
        pk = float(np.max(np.abs(x))) if x.size else 0.0
        sc = pk / 32767.0 if pk > 0 else 1.0 / 32768.0
        return True, sc, np.rint(x.astype(np.float64) / sc).astype(np.int16)
    return False, 1.0, x


class PoolIndex:
    """Grupos, pertença e a tabela compartilhada `posição -> (gid, slot físico, geração, comprimento)` (CPU; os workers
    do DataLoader herdam por fork e leem a memória compartilhada; só o processo principal escreve)."""

    def __init__(self, pools):
        paths, srs, bands = [], [], []
        groups: list[np.ndarray] = []
        self.grp_tier: dict[tuple[str, str], int] = {}
        self.spk_first: dict[str, tuple[int, int]] = {}
        for key in sorted(pools.recs):
            lst = pools.recs[key]
            g0 = len(paths)
            for pth, sr, band in lst:
                paths.append(pth)
                srs.append(int(sr))
                bands.append(float(band))
            self.spk_first[key] = (g0, len(lst))
            for tier, idx in pools.rec_tier[key].items():
                self.grp_tier[(key, tier)] = len(groups)
                groups.append(np.asarray(idx, np.int64) + g0)
        self.n_speech = len(paths)
        for pth, sr, band in pools.noise:
            paths.append(pth)
            srs.append(int(sr))
            bands.append(float(band))
        self.noise_group = len(groups)
        groups.append(np.arange(self.n_speech, len(paths), dtype=np.int64))
        self.paths = paths
        self.srs = np.asarray(srs, np.int64)
        self.bands = np.asarray(bands, np.float64)
        self.groups = groups
        self.N = np.asarray([len(g) for g in groups], np.int64)
        self.gid_group = np.empty(len(paths), np.int64)
        self.gid_rank = np.empty(len(paths), np.int64)
        for g, gl in enumerate(groups):
            self.gid_group[gl] = g
            self.gid_rank[gl] = np.arange(len(gl))
        self.k = np.zeros(len(groups), np.int64)
        self.ptr = np.zeros(len(groups) + 1, np.int64)
        self.tab = None
        self.flag = None
        self.pos_group = None

    def rec(self, gid: int) -> tuple[str, int, float]:
        return self.paths[gid], int(self.srs[gid]), float(self.bands[gid])

    def publish(self, members: list[list[tuple[int, int, int, int]]]):
        """`members[g]` = [(gid, phys, gen, len)] na ordem das posições; cria a tabela em memória compartilhada."""
        self.k = np.asarray([len(m) for m in members], np.int64)
        if np.any(self.k < 1):
            raise RuntimeError(f"pool: {int(np.sum(self.k < 1))} grupos sem clipe (orçamento pequeno demais)")
        self.ptr = np.concatenate([[0], np.cumsum(self.k)])
        P = int(self.ptr[-1])
        tab = torch.zeros(P, 2, 4, dtype=torch.int64)
        for g, m in enumerate(members):
            tab[self.ptr[g] : self.ptr[g + 1], 0] = torch.as_tensor(m, dtype=torch.int64)
        self.tab = tab.share_memory_()
        self.flag = torch.zeros(P, dtype=torch.int64).share_memory_()
        self.pos_group = np.repeat(np.arange(len(members)), self.k)

    def entry(self, pos: int) -> tuple[int, int, int, int]:
        f = int(self.flag[pos])
        e = self.tab[pos, f]
        return int(e[0]), int(e[1]), int(e[2]), int(e[3])

    def set_entry(self, pos: int, gid: int, phys: int, gen: int, length: int):
        other = 1 - int(self.flag[pos])
        self.tab[pos, other] = torch.as_tensor([gid, phys, gen, length], dtype=torch.int64)
        self.flag[pos] = other

    def draw_group(self, rng: np.random.Generator, g: int, r: int | None = None):
        """Posição uniforme entre os membros do grupo; grupo completo: a posição é o sorteio `r` do caminho original."""
        n = int(self.N[g])
        if r is None:
            r = int(rng.integers(n))
        k = int(self.k[g])
        p = r if k == n else int(rng.integers(k))
        return self.entry(int(self.ptr[g]) + p)

    def draw_speech(self, rng: np.random.Generator, key: str, tier: str | None):
        if tier:
            return self.draw_group(rng, self.grp_tier[(key, tier)])
        g0, n = self.spk_first[key]
        u = g0 + int(rng.integers(n))  # mesmo sorteio do `_speech` original: uniforme no falante
        return self.draw_group(rng, int(self.gid_group[u]), int(self.gid_rank[u]))

    def draw_noise(self, rng: np.random.Generator):
        return self.draw_group(rng, self.noise_group)


def group_rates(ds: StageV4RecipeDataset, index: PoolIndex, n_sim: int = 20000, seed: int = 12345) -> np.ndarray:
    """Taxa esperada de sorteios por grupo por amostra, simulando a parte de falantes da receita (sem áudio):
    usada só para repartir o orçamento (k_g ∝ taxa iguala o reuso por clipe), não altera a amostragem."""
    p = ds.p
    rate = np.zeros(len(index.groups))
    rng = np.random.default_rng(seed)
    npr = tuple(p.n_probs)
    for _ in range(n_sim):
        i = int(rng.integers(len(ds)))
        phase = ds.phase(i)
        tier = ds.slots[i % p.batch_size]
        probs = np.array(npr[:3]) / sum(npr[:3]) if phase == 1 else np.array(npr)
        n_int = int(rng.choice(len(probs), p=probs))
        p_hard = {1: 0.20, 2: 0.35, 3: 0.50}[phase]
        tk = ds.pools.target_tiers[tier]
        tkey = tk[int(rng.integers(len(tk)))]
        rate[index.grp_tier[(tkey, tier)]] += 1
        taken = {tkey}
        for _ in range(n_int):
            try:
                k = ds._pick_interferer(rng, tkey, taken, p_hard)
            except ValueError:
                break
            taken.add(k)
            g0, n = index.spk_first[k]
            for g in np.unique(index.gid_group[g0 : g0 + n]):
                rate[g] += float(np.sum(index.gid_group[g0 : g0 + n] == g)) / n
        rate[index.noise_group] += 0.7 if phase == 3 else 0.8
    return rate / n_sim


class GpuClipPool:
    """Arena na GPU + gerência (preenchimento inicial, renovação em segundo plano, quarentena, conferência de geração)."""

    def __init__(self, ds: StageV4RecipeDataset, device, budget_gb: float, *, codec: str = "exact",
                 refresh_pct_per_min: float = 0.0, delay_batches: int = 64, headroom: float = 0.08, seed: int = 0,
                 threads: int = 8, log=print):
        self.ds, self.dev, self.codec, self.log = ds, torch.device(device), codec, log
        self.index = PoolIndex(ds.pools)
        self.rng = np.random.default_rng([int(seed), 0x9001])
        self.n_pages = int(budget_gb * 1e9) // PAGE_BYTES
        self.arena = torch.zeros(self.n_pages * PAGE_BYTES, dtype=torch.uint8, device=self.dev)
        self.a16 = self.arena.view(torch.int16)
        self.a32 = self.arena.view(torch.float32)
        self.free_pages = list(range(self.n_pages - 1, -1, -1))
        self.headroom_pages = int(headroom * self.n_pages)
        self.delay = int(delay_batches)
        self.refresh_pct = float(refresh_pct_per_min)
        self.threads = int(threads)
        self.lock = threading.Lock()
        self.consumed = 0
        self.last_event = None
        self.quarantine: list[tuple[int, int]] = []  # (slot físico, lote a partir do qual pode ser solto)
        self.st = {"swaps": 0, "swap_skipped_full_group": 0, "swap_skipped_no_space": 0, "gen_mismatch_rows": 0,
                   "fill_s": 0.0}
        self._stop = threading.Event()
        self._th = None
        self._fill()

    # --- slots físicos -------------------------------------------------------------------------------------------
    def _alloc(self, n_slots: int):
        self.pt = torch.zeros(n_slots, WIDTH, dtype=torch.int64, device=self.dev)
        self.kind16 = torch.zeros(n_slots, dtype=torch.bool, device=self.dev)
        self.scale = torch.ones(n_slots, dtype=torch.float32, device=self.dev)
        self.gen = np.zeros(n_slots, np.int64)
        self.phys_pages: list[list[int]] = [[] for _ in range(n_slots)]
        self.phys_meta = np.zeros((n_slots, 3), np.int64)  # (gid, comprimento, é_int16)
        self.free_phys = list(range(n_slots - 1, -1, -1))

    def _load(self, gid: int) -> np.ndarray:
        return np.ascontiguousarray(self.ds._load48(self.index.paths[gid]), dtype=np.float32)

    def _write(self, gid: int, x: np.ndarray, reserve: int, stream=None) -> int | None:
        """Grava o clipe em páginas livres (mantendo `reserve` páginas) e devolve o slot físico, ou None sem espaço."""
        is16, sc, data = encode(x, self.codec)
        sp = S16 if is16 else S32
        npg = max(1, -(-len(data) // sp))
        if npg > WIDTH:
            raise ValueError(f"{self.index.paths[gid]}: {len(data) / SR:.1f} s > {WIDTH * S32 / SR:.0f} s por slot")
        with self.lock:
            if len(self.free_pages) - npg < reserve or not self.free_phys:
                return None
            pages = [self.free_pages.pop() for _ in range(npg)]
            phys = self.free_phys.pop()
        buf = np.zeros(npg * sp, data.dtype)
        buf[: len(data)] = data
        pg = torch.as_tensor(pages, dtype=torch.int64)
        ctx = torch.cuda.stream(stream) if stream is not None else _null()
        with ctx:
            src = torch.from_numpy(buf).to(self.dev, non_blocking=False)
            pgd = pg.to(self.dev)
            if is16:
                self.a16.view(self.n_pages, S16)[pgd] = src.view(npg, S16)
            else:
                self.a32.view(self.n_pages, S32)[pgd] = src.view(npg, S32)
            row = torch.zeros(WIDTH, dtype=torch.int64)
            row[:npg] = pg
            self.pt[phys] = row.to(self.dev)
            self.kind16[phys] = bool(is16)
            self.scale[phys] = float(sc)
        self.phys_pages[phys] = pages
        self.phys_meta[phys] = (gid, len(data), int(is16))
        return phys

    def _free(self, phys: int):
        with self.lock:
            self.free_pages.extend(self.phys_pages[phys])
            self.phys_pages[phys] = []
            self.gen[phys] += 1
            self.free_phys.append(phys)

    # --- preenchimento inicial ------------------------------------------------------------------------------------
    def _fill(self):
        t0 = time.monotonic()
        ix = self.index
        rate = group_rates(self.ds, ix)
        # bytes médios por clipe (amostra do manifest): fala e ruído
        def mean_bytes(gids):
            tot = 0.0
            for gid in gids:
                inf = sf.info(str(self.ds.data_root / ix.paths[gid]))
                lossless = inf.samplerate == SR and inf.subtype == "PCM_16"
                tot += inf.duration * SR * (2 if (lossless or self.codec == "int16") else 4)
            return tot / max(len(gids), 1)
        sp_b = mean_bytes(self.rng.choice(ix.n_speech, min(200, ix.n_speech), replace=False))
        nz = ix.groups[ix.noise_group]
        nz_b = mean_bytes(self.rng.choice(nz, min(100, len(nz)), replace=False))
        clip_b = np.full(len(ix.groups), sp_b)
        clip_b[ix.noise_group] = nz_b
        usable = (self.n_pages - self.headroom_pages) * PAGE_BYTES * 0.97
        lo, hi = 0.0, 1e9  # k_g = clamp(c·taxa, 1, N_g): busca de c para encher ~110% do orçamento (o excesso fica de fora)
        for _ in range(60):
            c = (lo + hi) / 2
            kg = np.clip(np.round(c * rate), 1, ix.N)
            if float(np.sum(kg * clip_b)) > 1.1 * usable:
                hi = c
            else:
                lo = c
        kg = np.clip(np.round(lo * rate), 1, ix.N).astype(np.int64)
        # ordem de carga: 1 clipe por grupo primeiro (obrigatório), depois o resto intercalado ao acaso
        chosen = [self.rng.permutation(ix.groups[g])[: kg[g]] for g in range(len(ix.groups))]
        first = [(g, int(c[0])) for g, c in enumerate(chosen)]
        rest = [(g, int(gid)) for g, c in enumerate(chosen) for gid in c[1:]]
        rest = [rest[j] for j in self.rng.permutation(len(rest))]
        order = first + rest
        self._alloc(int(np.sum(kg)) + max(256, int(0.05 * np.sum(kg))))
        members: list[list[tuple[int, int, int, int]]] = [[] for _ in ix.groups]
        full_stop = False
        with ThreadPoolExecutor(self.threads) as ex:
            win = 4 * self.threads
            for s in range(0, len(order), win):
                part = order[s : s + win]
                for (g, gid), x in zip(part, ex.map(lambda t: self._load(t[1]), part)):
                    reserve = self.headroom_pages if s >= len(first) else 0
                    phys = self._write(gid, x, reserve)
                    if phys is None:
                        if s < len(first):
                            raise RuntimeError("pool: orçamento não comporta 1 clipe por grupo")
                        full_stop = True
                        break
                    members[g].append((gid, phys, int(self.gen[phys]), int(self.phys_meta[phys, 1])))
                if full_stop:
                    break
        for g, m in enumerate(members):  # grupo completo: posições na ordem do grupo (posição = sorteio original)
            if len(m) == ix.N[g]:
                m.sort(key=lambda e: int(ix.gid_rank[e[0]]))
        ix.publish(members)
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)
        self.st["fill_s"] = time.monotonic() - t0
        self.log(f"pool GPU: {self.describe()}")

    def summary(self) -> dict:
        ix = self.index
        live = [int(ix.entry(p)[1]) for p in range(int(ix.ptr[-1]))]
        meta = self.phys_meta[live]
        sp = ix.pos_group != ix.noise_group
        secs = meta[:, 1] / SR
        used = self.n_pages - len(self.free_pages)
        return {"clips_speech": int(np.sum(sp)), "clips_noise": int(np.sum(~sp)),
                "hours_speech": float(np.sum(secs[sp]) / 3600), "hours_noise": float(np.sum(secs[~sp]) / 3600),
                "manifest_clips_speech": ix.n_speech, "manifest_clips_noise": int(ix.N[ix.noise_group]),
                "groups": len(ix.groups), "groups_full": int(np.sum(ix.k == ix.N)),
                "int16_frac_samples": float(np.sum(meta[:, 1] * meta[:, 2]) / max(np.sum(meta[:, 1]), 1)),
                "gb_used": used * PAGE_BYTES / 1e9, "gb_budget": self.n_pages * PAGE_BYTES / 1e9,
                "codec": self.codec, "refresh_pct_per_min": self.refresh_pct, **self.st}

    def describe(self) -> str:
        s = self.summary()
        return (f"{s['clips_speech']} clipes de fala ({s['hours_speech']:.1f} h de {s['manifest_clips_speech']}) + "
                f"{s['clips_noise']} de ruído ({s['hours_noise']:.1f} h de {s['manifest_clips_noise']}), "
                f"{s['gb_used']:.2f}/{s['gb_budget']:.2f} GB, int16 {100 * s['int16_frac_samples']:.0f}% das amostras, "
                f"codec {s['codec']}, {s['groups_full']}/{s['groups']} grupos completos, carga {s['fill_s']:.0f} s, "
                f"renovação {self.refresh_pct}%/min")

    # --- renovação ------------------------------------------------------------------------------------------------
    def start(self):
        if self.refresh_pct > 0 and self._th is None:
            self._th = threading.Thread(target=self._refresh_loop, name="gpu-pool-refresh", daemon=True)
            self._th.start()

    def close(self):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=10)

    def note_batch(self):
        """Chamado pelo misturador depois de enfileirar cada lote (conta lotes consumidos e guarda o evento)."""
        if self.dev.type == "cuda":
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.dev))
            self.last_event = ev
        self.consumed += 1

    def _release(self):
        due = [q for q in self.quarantine if q[1] <= self.consumed]
        if not due:
            return
        ev = self.last_event
        if ev is not None:
            ev.synchronize()  # a última mistura que pode ter lido os slots antigos terminou
        for q in due:
            self._free(q[0])
        self.quarantine = [q for q in self.quarantine if q[1] > self.consumed]

    def swap_once(self, stream=None) -> bool:
        ix = self.index
        P = int(ix.ptr[-1])
        pos = int(self.rng.integers(P))
        g = int(ix.pos_group[pos])
        if ix.k[g] >= ix.N[g]:
            self.st["swap_skipped_full_group"] += 1
            return False
        cur = {ix.entry(int(p))[0] for p in range(int(ix.ptr[g]), int(ix.ptr[g + 1]))}
        grp = ix.groups[g]
        while True:
            gid = int(grp[int(self.rng.integers(len(grp)))])
            if gid not in cur:
                break
        x = self._load(gid)
        phys = self._write(gid, x, 0, stream)
        if phys is None:
            self._release()
            phys = self._write(gid, x, 0, stream)
        if phys is None:
            self.st["swap_skipped_no_space"] += 1
            return False
        if stream is not None:
            stream.synchronize()
        old = ix.entry(pos)[1]
        ix.set_entry(pos, gid, phys, int(self.gen[phys]), int(self.phys_meta[phys, 1]))
        self.quarantine.append((old, self.consumed + self.delay))
        self.st["swaps"] += 1
        return True

    def _refresh_loop(self):
        stream = torch.cuda.Stream(self.dev) if self.dev.type == "cuda" else None
        per_s = self.refresh_pct / 100.0 * int(self.index.ptr[-1]) / 60.0
        acc, t = 0.0, time.monotonic()
        try:
            while not self._stop.wait(0.5):
                now = time.monotonic()
                acc += per_s * (now - t)
                t = now
                self._release()
                while acc >= 1.0 and not self._stop.is_set():
                    acc -= 1.0
                    self.swap_once(stream)
        except BaseException as e:  # noqa: BLE001
            self.log(f"pool GPU: renovação parou: {e!r}")

    # --- leitura (misturador) -------------------------------------------------------------------------------------
    def gather(self, ref: Tensor, n: int, wrap: bool, gid_col: int) -> Tensor:
        """`ref[M, ≥4]` = (slot físico, geração, início, comprimento, ...; gid na coluna `gid_col`) -> áudio f32
        `[M, n]` na GPU. `wrap=False`: recorte `crop_or_pad` (zeros depois do fim); `wrap=True`: `loop_to_length`."""
        M = ref.shape[0]
        out = torch.empty(M, n, dtype=torch.float32, device=self.dev)
        if M == 0:
            return out
        ref_c = ref.cpu()
        refd = ref_c.to(self.dev, non_blocking=True)
        t = torch.arange(n, device=self.dev)
        for s in range(0, M, 16):
            r = refd[s : s + 16]
            phys, start, length = r[:, 0], r[:, 2], r[:, 3]
            if wrap:
                src = (start[:, None] % length[:, None] + t[None]) % length[:, None]
            else:
                src = start[:, None] + t[None]
            valid = src < length[:, None]
            k16 = self.kind16[phys]
            sp = torch.where(k16, S16, S32)[:, None]
            page = torch.gather(self.pt[phys], 1, (src // sp).clamp(max=WIDTH - 1))
            idx = page * sp + src % sp
            v16 = self.a16[idx.clamp(max=self.a16.numel() - 1)].float() * self.scale[phys][:, None]
            v32 = self.a32[idx.clamp(max=self.a32.numel() - 1)]
            out[s : s + 16] = torch.where(valid, torch.where(k16[:, None], v16, v32), 0.0)
        bad = np.nonzero(self.gen[ref_c[:, 0].numpy()] != ref_c[:, 1].numpy())[0]
        for m in bad:  # slot reaproveitado antes da hora (não deveria ocorrer): recarrega do disco
            x = self._load(int(ref_c[m, gid_col]))
            rng = _FixedStart(int(ref_c[m, 2]))
            y = loop_to_length(x, n, rng) if wrap else crop_or_pad(x, n, rng)
            out[int(m)] = torch.from_numpy(np.asarray(y, np.float32)).to(self.dev)
        self.st["gen_mismatch_rows"] += len(bad)
        return out

    def floor_white(self, seeds: Tensor, n: int) -> Tensor:
        """Branco do piso (N(0,1) f32) de cada amostra silenciosa, gerado na GPU com a semente da receita."""
        out = torch.empty(len(seeds), n, dtype=torch.float32, device=self.dev)
        for j, s in enumerate(seeds.reshape(-1).tolist()):
            g = torch.Generator(device=self.dev)
            g.manual_seed(int(s))
            out[j] = torch.randn(n, generator=g, device=self.dev, dtype=torch.float32)
        return out


class _FixedStart:
    """Stand-in de `rng` para `crop_or_pad`/`loop_to_length` com o início já sorteado."""

    def __init__(self, start: int):
        self.start = start

    def integers(self, lo, hi=None):
        return self.start


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class StageV4PoolRecipeDataset(StageV4RecipeDataset):
    """Receitas que referenciam o pool residente em vez de carregar áudio (mesma sequência de sorteios do original;
    ver o docstring do módulo para a equivalência da lei de amostragem)."""

    def __init__(self, base: StageV4RecipeDataset, index: PoolIndex):
        self.__dict__.update(base.__dict__)
        self.index = index

    def _speech(self, rng, key: str, n: int, tier: str | None = None):
        gid, phys, gen, L = self.index.draw_speech(rng, key, tier)
        start = int(rng.integers(0, L - n + 1)) if L >= n else 0  # = crop_or_pad
        return np.asarray([phys, gen, start, L, 0, gid], np.int64), self.index.rec(gid)

    def _tilt(self, x):
        x = x.copy()
        x[4] = 1
        return x

    def _is_dead(self, x, on: np.ndarray, frames: int) -> bool:
        """Só o que se sabe sem o áudio: sem quadro ligado, ou todos os quadros ligados (±1) no enchimento de zeros
        depois do fim do clipe. Silêncio digital dentro do clipe a GPU detecta (`valid`) e a amostra é refeita com a
        tentativa seguinte, como o `_build` faria."""
        if not on.any():
            return True
        L = int(x[3])
        if L >= frames * HOP:
            return False
        dil = on.copy()
        dil[1:] |= on[:-1]
        dil[:-1] |= on[1:]
        return not bool(np.any(np.nonzero(dil)[0] * HOP < L))

    def _noise_draw(self, rng: np.random.Generator, n: int):
        gid, phys, gen, L = self.index.draw_noise(rng)
        reps = int(np.ceil(n / L)) + 1
        start = int(rng.integers(0, L * reps - n + 1))  # = loop_to_length (crop_or_pad do sinal repetido)
        return np.asarray([phys, gen, start, L, gid], np.int64)

    def _floor_draw(self, rng: np.random.Generator, n: int):
        return np.asarray([int(rng.integers(2**62))], np.int64)

    def _audio_fields(self, tracks: list, noise, floor_white) -> dict:
        return {"dry_ref": np.stack(tracks), "n_samples": np.int64(int(self.p.segment_s * SR)),
                "noise_ref": np.zeros(0, np.int64) if noise is None else noise,
                "floor_seed": np.zeros(0, np.int64) if floor_white is None else floor_white}

    def materialize(self, r: dict, pool: GpuClipPool) -> dict:
        """Receita do pool -> receita densa equivalente (áudio lido do disco como no caminho atual), para o teste de
        equivalência contra o `GpuMixer`; o branco do piso vem do mesmo gerador da GPU."""
        n = int(r["n_samples"])
        d = {k: v for k, v in r.items() if k not in ("dry_ref", "noise_ref", "floor_seed", "n_samples")}
        tracks = []
        for ref in r["dry_ref"]:
            x = crop_or_pad(pool._load(int(ref[5])), n, _FixedStart(int(ref[2])))
            if int(ref[4]):
                from cctrain.data.mix_v4 import whisper_tilt
                x = whisper_tilt(x)
            tracks.append(np.asarray(x, np.float32))
        d["dry"] = np.stack(tracks)
        nr = r["noise_ref"]
        d["noise"] = (loop_to_length(pool._load(int(nr[4])), n, _FixedStart(int(nr[2]))).astype(np.float32)
                      if len(nr) else np.zeros(0, np.float32))
        fs = r["floor_seed"]
        d["floor_white"] = pool.floor_white(torch.as_tensor(fs), n)[0].cpu().numpy() if len(fs) else np.zeros(0, np.float32)
        return d
