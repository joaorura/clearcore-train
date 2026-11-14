#!/usr/bin/env python3
"""Sonda T0 da `docs/proposta-arquitetura-tse.md` §3.1: "casamento nativo" — o cadastro como conjunto de quadros no
espaço de features do próprio DFNet3 (convs do encoder publicado, CONGELADAS, saída pré-GRU `combine(e3, df_fc_emb)`
[512]) e casamento quadro × cadastro (K tokens por pooling de atenção, ou todos os quadros), antes de qualquer GRU.

Mesmo dev do `ceiling_*` (304 receitas / 520 vistas do `eval_presence.py`, cache `dev-cache.npz` com rótulos, cosseno
ECAPA e perfis); mesmo treino das cabeças (mix_v4 em CPU, BCE com pesos por classe do `train_pro_v4_presence`).

Cadastro: falas do falante-alvo DIFERENTES da fala da mistura (treino: `pools.recs[alvo]` sem a fala sorteada nem as
reservadas; dev: `bank.utts[falante]` sem as falas usadas nos trechos do alvo da receita), concatenadas até 8 s;
quadros com VAD pico−30.

Variantes (mesmos lotes):
  nat_k16 / nat_k48      casamento com 16 / 48 tokens (pooling de atenção com consultas aprendidas)
  nat_all                casamento com todos os quadros ativos do cadastro (sem pooling)
  nat_k48_ecapa          nat_k48 + [cos ECAPA 1 s, cos·energia, energia] (como os `comb_*`)
  pre_ecapa              mesmas features pré-GRU, sem cadastro nativo: z ⊙ W_s·ECAPA (como a PresenceHead h32)

Uso:
  ceiling_native.py cache --dev-cache <dev-cache.npz> --out <native-dev.npz>
  ceiling_native.py train --dev-cache <dev-cache.npz> --native-dev <native-dev.npz> --run-dir <dir> [--steps 3500]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import ceiling_common as C
import eval_presence as E
import train_pro_v4_presence as P
import cctrain.data.mix_v4 as MV
from cctrain.data.features import extract_features
from cctrain.data.resample import to_16k
from cctrain.eval import tse_matrix as tm
from cctrain.paths import Paths

T = 500
NE = 800  # quadros do cadastro (8 s)
ENR_S = 8.0
HOP = 480


# --- features pré-GRU do DFNet3 publicado ------------------------------------------------------------------------------


def load_base(dev: str):
    from cctrain.models.dfnet3_upstream import load_upstream_dfnet3

    base, _ = load_upstream_dfnet3(device=dev)
    for p in base.parameters():
        p.requires_grad_(False)
    return base.eval()


@torch.no_grad()
def pre_gru(base, ferb: torch.Tensor, fspec: torch.Tensor) -> torch.Tensor:
    """ferb [B,1,T,32], fspec [B,1,T,96,2] -> [B,T,512] (pad_feat como no forward: índice t = quadro de saída t)."""
    fs = fspec.squeeze(1).permute(0, 3, 1, 2)
    ferb, fs = base.pad_feat(ferb), base.pad_feat(fs)
    enc = base.enc
    e3 = enc.erb_conv3(enc.erb_conv2(enc.erb_conv1(enc.erb_conv0(ferb))))
    c1 = enc.df_conv1(enc.df_conv0(fs))
    cemb = enc.df_fc_emb(c1.permute(0, 2, 3, 1).flatten(2))
    return enc.combine(e3.permute(0, 2, 3, 1).flatten(2), cemb)


def enroll_feats(x48: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Áudio do cadastro -> (feat_erb [1,NE,32], feat_spec [1,NE,96,2], ativo [NE]) com NE quadros (zero-pad)."""
    n = NE * HOP
    x = np.zeros(n, np.float32)
    m = min(n, len(x48))
    x[:m] = x48[:m]
    f = extract_features(x)
    act = MV.rel_activity(x.astype(np.float64), NE)
    act[m // HOP:] = False
    return np.asarray(f.feat_erb, np.float32), np.asarray(f.feat_spec, np.float32), act


def concat_enroll(paths: list[str], load, rng: np.random.Generator) -> np.ndarray:
    out, have = [], 0
    for j in rng.permutation(len(paths)):
        x = np.asarray(load(paths[int(j)]), np.float32)
        out.append(x)
        have += len(x)
        if have >= ENR_S * 48_000:
            break
    return np.concatenate(out) if out else np.zeros(1, np.float32)


# --- cache do dev ------------------------------------------------------------------------------------------------------


def cmd_cache(a):
    dev = a.device
    paths = Paths.from_env()
    sd = paths.data / "eval-v4"
    tm.set_roots(paths.data, sd / "user_take4_stream_48k.wav")
    bank = tm.Bank.from_json(json.loads((sd / "bank.json").read_text()), np.load(sd / "bank_emb.npy"))
    recs = E.select(tm.read_recipes(sd / "recipes.jsonl"))
    old = np.load(a.dev_cache)
    base = load_base(dev)
    xs, enr_idx, enr_x, enr_m, ids = [], [], [], [], []
    t0 = time.time()
    for k, rec in enumerate(recs):
        r = tm.render(rec)
        c = rec["cell"]
        spk = rec["profile"]["spk"] if rec["profile"]["kind"] == "bank" else None
        if spk is None:
            raise SystemExit("perfil 'user' no dev: não previsto")
        used = {s["path"] for s in rec["target"]["segs"] if s.get("path")}
        pool = [u["path"] for u in bank.utts[spk] if u["path"] not in used] or [u["path"] for u in bank.utts[spk]]
        xe = concat_enroll(pool, tm.load48, np.random.default_rng(k))
        fe, fs, act = enroll_feats(xe)
        t = lambda z: torch.from_numpy(np.asarray(z))[None].to(dev)  # noqa: E731
        enr_x.append(pre_gru(base, t(fe), t(fs))[0].half().cpu().numpy())
        enr_m.append(act)
        views = [("mix", r.noisy(True))]
        if c["n"] >= 1 and c["axis"] != "temporal":
            views.append(("int", r.noisy(False)))
        for vname, x in views:
            f = extract_features(x)
            xp = pre_gru(base, t(f.feat_erb), t(f.feat_spec))[0, :T]
            xs.append(xp.half().cpu().numpy())
            enr_idx.append(len(enr_x) - 1)
            ids.append(rec["id"])
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(recs)} receitas ({time.time() - t0:.0f} s)", flush=True)
    assert ids == [str(i) for i in old["id"]], "ordem das vistas difere do dev-cache"
    np.savez(a.out, xpre=np.stack(xs), enr_idx=np.asarray(enr_idx), enr_x=np.stack(enr_x), enr_m=np.stack(enr_m))
    print(f"{len(ids)} vistas, {len(enr_x)} cadastros -> {a.out} ({time.time() - t0:.0f} s)")


# --- cabeças -----------------------------------------------------------------------------------------------------------


class NativeHead(nn.Module):
    """Casamento quadro × cadastro no espaço pré-GRU (siamês: mesma projeção f para mistura e cadastro).

    tokens = pooling de atenção do cadastro (K consultas aprendidas; K=0 = todos os quadros ativos)
    s_t = [max_k cos, média cos, soft-max cos (τ 0,1), desvio cos]; c_t = atenção de q(f(x_t)) sobre os tokens
    entrada da GRU = [s_t, relu(W x_t), W_c (c_t ⊙ f(x_t)), extra] -> GRU -> logit."""

    def __init__(self, K: int = 48, d: int = 128, proj: int = 64, hidden: int = 64, extra: int = 0,
                 init_p: float = 0.9):
        super().__init__()
        self.K, self.d = K, d
        self.norm = nn.LayerNorm(512)
        self.f = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, d))
        if K > 0:
            self.tok_q = nn.Parameter(torch.randn(K, d) * d ** -0.5)
            self.tok_k = nn.Linear(d, d)
        self.q = nn.Linear(d, d)
        self.inp = nn.Linear(512, proj)
        self.ctx = nn.Linear(d, proj)
        self.gru = nn.GRU(4 + 2 * proj + extra, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)
        nn.init.constant_(self.out.bias, math.log(init_p / (1 - init_p)))

    def tokens(self, xe: torch.Tensor, me: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fe = self.f(self.norm(xe))  # [B,Ne,d]
        if self.K == 0:
            return fe, me
        att = torch.einsum("kd,bnd->bkn", self.tok_q, self.tok_k(fe)) / math.sqrt(self.d)
        att = att.masked_fill(~me[:, None, :], -1e4).softmax(-1)
        return att @ fe, torch.ones(fe.shape[0], self.K, dtype=torch.bool, device=fe.device)

    def forward(self, x, xe, me, extra=None):
        xn = self.norm(x)
        fx = self.f(xn)
        tok, tm_ = self.tokens(xe, me)  # [B,K,d], [B,K]
        cos = torch.einsum("btd,bkd->btk", F.normalize(fx, dim=-1), F.normalize(tok, dim=-1))
        neg = (~tm_)[:, None, :]
        cmax = cos.masked_fill(neg, -2.0).amax(-1)
        cnt = tm_.sum(-1).clamp(min=1)[:, None].float()
        cz = cos.masked_fill(neg, 0.0)
        cmean = cz.sum(-1) / cnt
        cstd = ((cz - cmean[..., None]).pow(2).masked_fill(neg, 0.0).sum(-1) / cnt).clamp(min=1e-8).sqrt()
        csoft = (cos / 0.1).masked_fill(neg, -1e4).softmax(-1).mul(cos).sum(-1)
        w = torch.einsum("btd,bkd->btk", self.q(fx), tok) / math.sqrt(self.d)
        c = w.masked_fill(neg, -1e4).softmax(-1) @ tok
        parts = [torch.stack([cmax, cmean, csoft, cstd], -1), torch.relu(self.inp(xn)), self.ctx(c * fx)]
        if extra is not None:
            parts.append(extra)
        y, _ = self.gru(torch.cat(parts, -1))
        return self.out(y).squeeze(-1)


class PreEcapaHead(nn.Module):
    """Linha de base nas MESMAS features pré-GRU: z = relu(W·x), s = W_s·ECAPA, GRU([z, z⊙s])."""

    def __init__(self, proj: int = 64, hidden: int = 64, init_p: float = 0.9):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.inp = nn.Linear(512, proj)
        self.spk = nn.Linear(192, proj)
        self.gru = nn.GRU(2 * proj, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)
        nn.init.constant_(self.out.bias, math.log(init_p / (1 - init_p)))

    def forward(self, x, emb):
        z = torch.relu(self.inp(self.norm(x)))
        y, _ = self.gru(torch.cat([z, z * self.spk(emb)[:, None]], -1))
        return self.out(y).squeeze(-1)


SPECS = {"nat_k16": dict(K=16), "nat_k48": dict(K=48), "nat_all": dict(K=0), "nat_k48_ecapa": dict(K=48, extra=3)}


# --- treino ------------------------------------------------------------------------------------------------------------


_LAST: dict = {}
_orig_extract = MV.extract_features


def _capture(x):
    _LAST["x"] = x
    return _orig_extract(x)


class DS(MV.StageV4Dataset):
    def __getitem__(self, i):
        d = super().__getitem__(i)
        x = _LAST["x"]
        n = len(x) // HOP
        d["x16"] = to_16k(x, 48_000)[: n * 160].astype(np.float32)
        fr = x[: n * HOP].astype(np.float64).reshape(n, HOP)
        d["mix_db"] = (10 * np.log10((fr * fr).mean(1) + 1e-12)).astype(np.float32)
        tkey, tpath = d.pop("target_key"), d.pop("target_path")
        excl = {tpath, *self.pools.heldout_enroll}
        pool = [r[0] for r in self.pools.recs[tkey] if r[0] not in excl] or [tpath]
        rng = np.random.default_rng([self.p.seed, i, 77])
        xe = concat_enroll(pool, self._load48, rng)
        d["enr_ferb"], d["enr_fspec"], d["enr_act"] = enroll_feats(xe)
        for k in ("spec_clean", "spec_interf"):
            d.pop(k, None)
        return d


def dev_auc_switch(c: dict, s: np.ndarray, win: int = 20) -> float:
    """AUC nos `win` quadros após troca de falante: alvo ativo com interferente ativo nos últimos `win` quadros
    (e agora não) × só-interferente com alvo ativo nos últimos `win` quadros (e agora não)."""
    ta, ia, valid = c["ta"], c["ia"], c["valid"]
    mix = (c["view"] == "mix")[:, None]

    def recent(m):
        out = np.zeros_like(m)
        for k in range(1, win + 1):
            out[:, k:] |= m[:, :-k]
        return out
    pos = ta & ~ia & recent(ia) & valid & mix
    neg = ~ta & ia & recent(ta) & valid & mix
    if not pos.any() or not neg.any():
        return float("nan")
    return C.auc(np.concatenate([s[pos], s[neg]]), np.concatenate([np.ones(pos.sum(), bool), np.zeros(neg.sum(), bool)]))


def cmd_train(a):
    MV.extract_features = _capture
    MV.analysis = lambda x: np.zeros(1, np.float32)
    from cctrain.m1 import pools
    from cctrain.train.embed_bank import ArrayEmbeddingBank
    from cctrain.train.stage_c import _to_device

    dev = a.device
    torch.manual_seed(a.seed)
    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    paths = Paths.from_env()
    base = load_base(dev)
    fbank, ecapa = C.load_ecapa(paths.data, dev)
    for p in ecapa.parameters():
        p.requires_grad_(False)
    heads: dict[str, nn.Module] = {k: NativeHead(**v).to(dev) for k, v in SPECS.items()}
    heads["pre_ecapa"] = PreEcapaHead().to(dev)
    nparams = {k: sum(p.numel() for p in h.parameters()) for k, h in heads.items()}
    print("parâmetros:", nparams, flush=True)
    params = [p for h in heads.values() for p in h.parameters()]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    vp = MV.V4Params(seed=a.seed, batch_size=a.batch_size, total_steps=a.steps, phase1_frac=0.0,
                     n_probs=(0.35, 0.30, 0.17, 0.11, 0.07))
    bank = ArrayEmbeddingBank.load(paths.data / "embeddings" / "bank-m2-sbecapa-v2.npz")
    speech = pools.by_split(pools.speech_records(paths), "train")
    noise_train, _ = pools.noise_pools()
    pl = MV.build_pools(speech, noise_train, pools.rir_records(), bank, paths.data, seed=vp.seed)
    ds = DS(pl, bank, paths.data, vp)
    dl = DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers, prefetch_factor=2, shuffle=False)

    # dev
    z = np.load(a.dev_cache)
    c = {k: z[k] for k in z.files if k != "emb31"}
    nd = np.load(a.native_dev)
    dt = {"x": torch.from_numpy(nd["xpre"]).to(dev), "enr_x": torch.from_numpy(nd["enr_x"]).to(dev),
          "enr_m": torch.from_numpy(nd["enr_m"]).to(dev), "enr_idx": torch.from_numpy(nd["enr_idx"]).to(dev),
          "prof": torch.from_numpy(c["profiles"]).to(dev),
          "extra": C.extra_feats(torch.from_numpy(c["cos1.0"]).float().to(dev),
                                 torch.from_numpy(c["mix_db"]).float().to(dev))}

    @torch.no_grad()
    def dev_scores(k, h):
        out = []
        for i in range(0, dt["x"].shape[0], 32):
            x = dt["x"][i: i + 32].float()
            if k == "pre_ecapa":
                lg = h(x, dt["prof"][i: i + 32])
            else:
                ei = dt["enr_idx"][i: i + 32]
                lg = h(x, dt["enr_x"][ei].float(), dt["enr_m"][ei],
                       dt["extra"][i: i + 32] if SPECS[k].get("extra") else None)
            out.append(torch.sigmoid(lg).float().cpu().numpy())
        return np.concatenate(out)

    curve = []
    t0 = time.time()
    step = 0
    run_loss = {k: 0.0 for k in heads}
    logf = open(run / "train-log.jsonl", "w")

    def do_eval(final=False):
        row = {"step": step, "elapsed_s": round(time.time() - t0, 1)}
        for k, h in heads.items():
            h.eval()
            s = dev_scores(k, h)
            pos, neg = C.pos_neg(c)
            sc = np.concatenate([s[pos], s[neg]])
            y = np.concatenate([np.ones(pos.sum(), bool), np.zeros(neg.sum(), bool)])
            row[k] = {"auc": round(C.auc(sc, y), 4), "auc_switch200": round(dev_auc_switch(c, s), 4)}
            if final:
                r = C.evaluate_score(c, s, groups=False)
                row[k].update({"ba_max": round(r["ba_max"], 4), "gate": r["gate"]})
                np.save(run / f"dev-scores-{k}.npy", s.astype(np.float16))
            h.train()
        curve.append(row)
        print(json.dumps(row), flush=True)

    for b in dl:
        if step >= a.steps:
            break
        b = _to_device(b, dev)
        with torch.no_grad():
            tb = b["mix_db"].shape[1]
            x = pre_gru(base, b["feat_erb"], b["feat_spec"])[:, :tb].float()
            xe = pre_gru(base, b["enr_ferb"], b["enr_fspec"]).float()
            me = b["enr_act"].bool()
            e = b["embedding"].float()
            e = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            cos1 = C.cos_per_frame(C.window_embeddings(fbank, ecapa, b["x16"], 1.0), e, tb)
            cos1 = torch.nan_to_num(cos1, nan=0.0, posinf=0.0, neginf=0.0)
            extra = C.extra_feats(cos1, b["mix_db"])
        ta, ia = b["target_active"][:, :tb], b["interf_active"][:, :tb]
        valid = ~b["digital_silence"][:, :tb]
        n0t = (b["n_interf"] == 0)[:, None] & ta
        total = 0.0
        for k, h in heads.items():
            lg = h(x, e) if k == "pre_ecapa" else h(x, xe, me, extra if SPECS[k].get("extra") else None)
            loss = P.bce_presence(lg, ta, ia, valid, n0t, 1.0)
            run_loss[k] = 0.98 * run_loss[k] + 0.02 * float(loss.detach()) if step else float(loss.detach())
            total = total + loss
        opt.zero_grad(set_to_none=True)
        total.backward()
        for h in heads.values():
            torch.nn.utils.clip_grad_norm_(h.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            ent = {"step": step, "s": round(time.time() - t0, 1), **{k: round(v, 4) for k, v in run_loss.items()}}
            logf.write(json.dumps(ent) + "\n")
            logf.flush()
            if step % 250 == 0:
                print(json.dumps(ent), flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            do_eval(final=step == a.steps)
    torch.save({k: h.state_dict() for k, h in heads.items()}, run / "heads.pt")
    (run / "curve.json").write_text(json.dumps({"params": nparams, "curve": curve, "args": vars(a)}, indent=1))
    print(f"fim: {step} passos em {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cache")
    p.add_argument("--dev-cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("train")
    p.add_argument("--dev-cache", required=True)
    p.add_argument("--native-dev", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--steps", type=int, default=3500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=20261113)
    p.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cmd_cache(a) if a.cmd == "cache" else cmd_train(a)


if __name__ == "__main__":
    main()
