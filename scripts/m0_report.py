"""I2 no asset FiLM gerado pelo repo + relatório M0 (docs/m0-report.md e runs/m0/report.json)."""

import json
import shutil

import yaml

from cctrain import tractcheck
from cctrain.data.manifest import FILES_DIR
from cctrain.data.resample import V1_DIR
from cctrain.eval.gates import load_budget, load_gates
from cctrain.export.film import build_film_identity_asset
from cctrain.m0.report import go_no_go, project_disk, render_markdown
from cctrain.paths import Paths, clearcore_root

PATHS = Paths.from_env()
RUNS = PATHS.runs / "m0"

def rir_summary() -> dict:
    """Faixa de RT60 medido e cap de ordem lidos dos parâmetros das RIR do piloto."""
    rows = [json.loads(line) for line in
            (FILES_DIR / "rir-synthetic-v1-pilot.params.jsonl").read_text().splitlines() if line.strip()]
    rt60 = [r["rt60_measured"] for r in rows]
    return {"count": len(rows), "rt60_min": min(rt60), "rt60_max": max(rt60),
            "rt60_median": sorted(rt60)[len(rt60) // 2], "max_order_cap": max(r["max_order"] for r in rows)}


def _pt(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def decisoes(offline: dict, rir: dict) -> str:
    return f"""\
- **Common Voice:** o Common Voice 27.0 pt (`cv27-pt`, filtro pt-BR "Portuguese (Brasil)" por metadado) substitui o
  26.0: as versões são cumulativas e o 26.0 exigia termos separados (decisão do orquestrador, não medido aqui).
- **Ruído do piloto:** os 3 ruídos CC0 do Clearcore são split `test` (não treináveis). O ruído treinável do piloto é
  sintético (`noise-synthetic-v1`, `LicenseRef-cctrain-generated`) e serve só para medir custo (decisão do
  orquestrador, não medido aqui).
- **Desvio emulação offline × runtime:** informativo no M0, sem gate (o `budget-v1` está congelado por SHA-256). Valores
  medidos em `offline_vs_runtime.json`: erro/sinal de {_pt(offline['error_to_signal_db'])} dB e
  {_pt(offline['stage_agreement'] * 100)}% de concordância de estágio (silêncio fora da conta:
  {_pt(offline.get('silent_fraction_excluded', 0.0) * 100)}% dos quadros). Alimenta a definição do gate antes do M2.
- **RIR `synthetic-v1`:** RT60 medido de {_pt(rir['rt60_min'], 2)} a {_pt(rir['rt60_max'], 2)} s (mediana
  {_pt(rir['rt60_median'], 2)} s) em {rir['count']} RIR do piloto, com `max_order` limitado a
  {rir['max_order_cap']} (lido de `rir-synthetic-v1-pilot.params.jsonl`). A cauda longa não existe: aceito no M0
  porque o piloto mede custo, não qualidade (decisão do orquestrador).
- **Mistura:** convenção de SNR (ruído escalado contra alvo + interferente) e alvo seco não alinhado à RIR aceitos no
  M0, a conferir contra a spec §4.4 (decisão do orquestrador, não medido aqui).
- **Splits e loader:** `training_records` recusa `test`/`dev`; o loader tem `allowed_splits`.
- **Go/no-go:** segue estritamente os níveis congelados de `configs/m0/budget-v1.yaml` (verificados por SHA-256);
  nenhum nível foi ajustado depois das medições.
"""


PROXIMOS = """\
Pendências **obrigatórias do M1** (o plano do M1 deve começar por elas):

1. Baixar ruído real de treino: Freesound CC0 do DNS e FSD50K (CC0/CC BY).
2. Gerar RIR `synthetic-v2` com cauda longa (o v1 cobre só RT60 de 0,1 a 0,6 s).
3. Conferir a convenção de SNR (ruído escalado contra alvo + interferente) e o alinhamento do alvo seco × reverberado
   contra a spec §4.4.
4. Definir o gate para o desvio emulação offline × runtime antes do M2.
5. Publicar o commit `70dc85c` do Clearcore: a dependência git do `tract-check` está pinada nele.
6. Triagem dos *deferred minors* do ledger (lista abaixo).
7. Re-projetar o disco incluindo o que o M0 deixou fora da projeção (RIR `synthetic-v2` com cauda longa, ambiente
   `export` do M3, DNS5 dev do G8, pico transitório de download/extração do CV27 e do FSD50K) antes do M1.

O que cada marco seguinte consome deste relatório está nos "Próximos planos" do plano bootstrap + M0 (M1: pooling que
passou no tract, parâmetros do fbank, fatores FLAC por SR e vazão do loader; M2: B fixado, s/passo e VRAM, desvio
offline × runtime; M3: asset reproduzível e I2; M4: réplica do gating e projeção de disco; M5: pacote e proveniência).

*Deferred minors* do ledger, para triagem:

| Tarefa | Minor |
|---|---|
| 2 | ruído "invalid character #" de fixtures do workspace-policy do Clearcore no build; fallback de `binary()` aponta para `repo_root` em worktree; toolchain só fixa via cwd |
| 3 | JSONL com U+2028/U+2029 quebra `splitlines`; `_read_lock` não converte `JSONDecodeError`; sr/num_samples sem validação de tipo; ruídos Wikimedia com role train split test entrariam em `training_records` |
| 4 | sem `.gitattributes` marcando `resampler/**` como binário (risco autocrlf no hash); coeficientes custom não validam shape 241; testes bit-exatos de `golden_inputs` dependem da plataforma |
| 5 | `verify_frozen` com lock vazio dá `IndexError`; `load_budget` sem validação de chaves; dataclasses frozen com dicts mutáveis; hash não fixado literalmente em teste; `test_budget_is_frozen` não prova quebra por edição |
| 6 | `verify_film_graph` não exige que todo consumidor do site leia `_film` nem valida forma/tipo de gamma/beta; `add_film` muta o `ModelProto` recebido; `_initializers` ignora sparse/Constant; `pack.py` deixa `.tmp` órfão em falha; nível zlib não fixado; testes de recusa de `read_members`/`pack_asset` só cobrem bit executável |
| 7 | `validate()` não checa embedding_dim/mid_dim; teste do Γ=ln10 não asserta o valor; limite Γ=ln100 não testado; regra de B duplicada entre `film.py` e YAML |
| 8 | I1 não cobre o ramo `lsnr_dropout=True`; `film_sites` não prova que a modulação ocorre após o ReLU; `state_hash` falha com bfloat16 |
| 9 | prototype com BN em estatísticas de inicialização (P1 não exercita BN não trivial); `scale_to_*` dividem por zero com sinal nulo |
| 10 | `requests.HTTPError` de `raise_for_status` e erros do RemoteZip podem expor URL pré-assinada completa |
| 11 | NaN em `rt60_measured` geraria JSON inválido; teste de determinismo não compara sha256 do FLAC |
| 12 | `pilot_records(librispeech_speakers=...)` ficou morto; `_chunks` não faz retry em ConnectionError/timeout |
| 13 | testes do loader só pegam o 1º item; `pilot_records` sem guarda para speaker None nem prefixo de source_id; SIR espúrio em INTERFERER_ONLY; ruído carregado sem uso em TARGET_INTERFERER; seed repetido por época |
| 14 | teste de gating não confere estágio contra limiares; lsnr NaN vira `erb_df` silenciosamente; cauda parcial descartada; tool mais estrito que o runtime com saídas extras não-f32 |
| infra | worktree novo não tem `build/` (basetemp do pytest) e `env.sh` sobrescreve PYTHONPATH: implementadores precisam `mkdir -p build` e reexportar `CCTRAIN_WT`/`PYTHONPATH` |
"""


def _load(name: str) -> dict:
    p = RUNS / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else {"status": "não executado"}


def i2_own() -> dict:
    root = clearcore_root()
    asset = build_film_identity_asset(root / "vendor/approved/df-compatible-release-asset-v1.bin", PATHS.build / "m0")
    out = RUNS / "i2_own.json"
    proc = tractcheck.run("identity", asset.path, root / "vendor/approved/df-compatible-release-asset-v1.bin",
                          root / "fixtures/film/upstream-golden-1000f.f32", out)
    rep = json.loads(out.read_text())
    rep.update(asset_sha256=asset.sha256, asset_size=asset.size, returncode=proc.returncode)
    out.write_text(json.dumps(rep, indent=2) + "\n")
    return rep


def fmt(obj) -> str:
    return "```json\n" + json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n```"


DISK_FORA = (
    "RIR `synthetic-v2` com cauda longa (a projeção conta as 10 000 RIR do `synthetic-v1`, curtas)",
    "ambiente `export` do M3 (Python 3.10, torch 1.13.1)",
    "DNS5 dev do G8",
    "pico transitório de download/extração do CV27 e do FSD50K (arquivos comprimidos e extraídos coexistem)",
)


def disk_note(disk_row: dict) -> str:
    margin = disk_row["limit"] - disk_row["value"]
    fora = "\n".join(f"- {x}" for x in DISK_FORA)
    return (f"O veredito de disco (**{disk_row['verdict']}**) tem margem de {margin:.2f} GB "
            f"(limite {disk_row['limit']:.2f} GB menos projeção {disk_row['value']:.2f} GB), "
            f"já descontada a reserva de 10 GB. **Não estão na projeção:**\n\n{fora}\n\n"
            "Por isso o go de disco é condicional: re-projetar o disco incluindo esses itens antes do M1.")


def limitacoes(runs: dict) -> str:
    env, loader = runs["env"], runs["loader"]
    b16, b32, tea = runs["step_b16"], runs["step_b32"], runs["step_b16_teacher"]
    steps = sorted({r["steps"] for r in (b16, b32, tea)})
    throttle = sorted({x for r in (b16, b32, tea) for x in r["gpu"]["nonzero_throttle_reasons"]})
    return "\n".join([
        "- **Perda proxy e professor stand-in:** o passo de GPU mede o custo com uma perda proxy, não a perda real "
        "do `df/loss.py`, e o professor é um stand-in; s/passo e VRAM com professor são ordem de grandeza "
        "(a perda real e o professor real entram no M2).",
        f"- **Sem estabilização térmica:** {', '.join(map(str, steps))} passos por configuração, curtos demais para o "
        f"estado estacionário; temperatura máxima observada {b32['gpu']['max_temperature_c']:.0f} C em B32.",
        f"- **Throttle:** razões não nulas de throttle em todas as medições ({', '.join(throttle)}); "
        f"`min_sm_clock` de {b32['gpu']['min_sm_clock_mhz']:.0f} MHz em B32 contra "
        f"{b16['gpu']['min_sm_clock_mhz']:.0f} MHz em B16 sem professor "
        f"({tea['gpu']['min_sm_clock_mhz']:.0f} MHz em B16 com professor): s/passo em B32 pode estar inflado.",
        f"- **Memória do host:** durante as medições havia {env['ram_available_gb']:.2f} GB de RAM disponível "
        f"(de {env['ram_total_gb']:.1f} GB) e {env['swap_used_gb']:.1f} GB de swap em uso; isso pode afetar a vazão "
        "do loader.",
        f"- **Loader:** medido com o ruído sintético do piloto, pequeno o bastante para caber no page cache; a vazão "
        f"de {loader['samples_per_s']:.0f} amostras/s e o RSS de {loader['rss_max_gb']:.2f} GB não valem para "
        "dados de treino reais maiores que a RAM.",
    ])


def ecapa_line(ecapa: dict) -> str:
    parts = [f"{m}: {'aprovado' if v.get('all_pass') else 'REPROVADO'}" for m, v in ecapa.items()]
    return "- ECAPA no tract (pooling): " + "; ".join(parts)


def main() -> None:
    i2 = i2_own()
    runs = {n: _load(n) for n in ("env", "i1", "i2_clearcore_fixture", "film_asset", "gating_parity",
                                  "offline_vs_runtime", "ecapa_tract", "disk", "loader", "step_b16", "step_b32",
                                  "step_b16_teacher", "beta_rule")}
    gates, budget = load_gates(), load_budget()
    minimal = yaml.safe_load((PATHS.repo / "configs/data/minimal-set-v1.yaml").read_text())
    proj = project_disk(minimal, runs["disk"])
    already_gb = sum(v or 0 for k, v in runs["disk"]["dirs_bytes"].items() if k in ("venv", "tract_check_target",
                                                                                     "rustup", "data")) / 1e9
    free_gb = shutil.disk_usage(PATHS.data).free / 1e9 + already_gb
    verdicts = go_no_go(budget, proj["total_gb"], free_gb, runs["step_b16_teacher"], runs["loader"])
    note = disk_note(verdicts[0])
    resampler = json.loads((V1_DIR / "manifest.json").read_text())["coefficients"]
    env = {k: v for k, v in runs["env"].items() if k != "nvidia_smi_q_head"}
    disk = {k: v for k, v in runs["disk"].items() if k != "tract_check_target_root"}
    margin = verdicts[0]["limit"] - verdicts[0]["value"]
    resumo = [f"- {r['item']}: **{r['verdict']}**" + (f" (margem {margin:.2f} GB; ver as exclusões em "
                                                      f"\"Go/no-go do orçamento\")" if r["item"] == "disco" else "")
              for r in verdicts]
    resumo += [f"- I1: **{'aprovado' if runs['i1'].get('pass') else 'REPROVADO'}**; "
               f"I2 (asset do repo): **{'aprovado' if i2.get('pass') else 'REPROVADO'}**",
               f"- B fixado: {runs['beta_rule'].get('beta_bound')}",
               f"- Réplica do gating: {'paridade ok' if runs['gating_parity'].get('pass') else 'FALHOU'}",
               ecapa_line(runs["ecapa_tract"])]
    sections = [
        ("Resumo", "\n".join(resumo)),
        ("Decisões do orquestrador", decisoes(runs["offline_vs_runtime"], rir_summary())),
        ("Ambiente", fmt(env)),
        ("Identidade (I1, I2)", fmt({"i1": runs["i1"], "i2_fixture_clearcore": runs["i2_clearcore_fixture"],
                                     "i2_asset_do_repo": i2, "film_asset": runs["film_asset"]})),
        ("Congelamentos", fmt({"gates_v1_sha256": gates.sha256, "budget_v1_sha256": budget.sha256,
                               "resampler_v1": resampler})),
        ("Réplica do gating × tract-check", fmt(runs["gating_parity"])),
        ("Emulação offline × runtime (dado para o M2)", fmt(runs["offline_vs_runtime"])),
        ("Protótipo fbank + ECAPA no tract", fmt({m: {"all_pass": v.get("all_pass"),
                                                     "falhas": [c for c in v.get("cases", []) if not c["passed"]]}
                                                 for m, v in runs["ecapa_tract"].items()})),
        ("Disco (fatores medidos e projeção do conjunto mínimo)", fmt({"medido": disk, "projecao": proj,
                                                                       "livre_mais_ja_usado_gb": free_gb,
                                                                       "ja_usado_gb": already_gb})),
        ("Loader", fmt(runs["loader"])),
        ("Passo de treino na GPU", fmt({k: runs[k] for k in ("step_b16", "step_b32", "step_b16_teacher")})),
        ("Regra de B", fmt(runs["beta_rule"])),
        ("Limitações das medições", limitacoes(runs)),
        ("Pendências", "- CV 27.0 pt: " + ("manifest `cv27-pt.jsonl` presente"
                                            if (PATHS.data / "manifests/cv27-pt.jsonl").is_file()
                                            else "não ingerido no piloto (custo medido pelos fatores por SR)")
         + "\n- Disco: fatores FLAC de fala por fonte (VCTK a 48 kHz, LibriSpeech dev-clean a 16 kHz, demais SR "
           "interpolados); ruído pelo agregado por SR; `tract_check_target` medido no checkout principal."
         + "\n- Decisões de planejamento 1-15 do plano bootstrap+M0 valem como registradas lá."),
        ("Próximos passos", PROXIMOS),
    ]
    md = render_markdown({"title": "Clearcore Train — relatório M0 (piloto)", "sections": sections,
                          "go_no_go": verdicts, "go_no_go_note": note})
    (PATHS.repo / "docs").mkdir(exist_ok=True)
    (PATHS.repo / "docs/m0-report.md").write_text(md)
    (RUNS / "report.json").write_text(json.dumps({"go_no_go": verdicts, "disk_projection": proj, "i2_own": i2,
                                                  "free_gb": free_gb, "already_gb": already_gb},
                                                 indent=2, default=str) + "\n")
    print(md)


if __name__ == "__main__":
    main()
