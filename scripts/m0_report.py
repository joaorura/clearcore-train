"""I2 no asset FiLM gerado pelo repo + relatório M0 (docs/m0-report.md e runs/m0/report.json)."""

import json
import shutil

import yaml

from cctrain import tractcheck
from cctrain.data.resample import V1_DIR
from cctrain.eval.gates import load_budget, load_gates
from cctrain.export.film import build_film_identity_asset
from cctrain.m0.report import go_no_go, project_disk, render_markdown
from cctrain.paths import Paths, clearcore_root

PATHS = Paths.from_env()
RUNS = PATHS.runs / "m0"

DECISOES = """\
- **Common Voice:** o Common Voice 27.0 pt (`cv27-pt`, filtro pt-BR "Portuguese (Brasil)" por metadado) substitui o
  26.0: as versões são cumulativas e o 26.0 exigia termos separados.
- **Ruído do piloto:** os 3 ruídos CC0 do Clearcore são split `test` (não treináveis). O ruído treinável do piloto é
  sintético (`noise-synthetic-v1`, `LicenseRef-cctrain-generated`) e serve só para medir custo.
- **Desvio emulação offline × runtime:** informativo no M0, sem gate (o `budget-v1` está congelado por SHA-256). O
  valor medido (erro/sinal de -17,5 dB; 92,6% de concordância de estágio) alimenta a definição do gate antes do M2.
- **RIR `synthetic-v1`:** RT60 efetivo de cerca de 0,1 a 0,6 s (cap de ordem 20), aceito no M0 porque o piloto mede
  custo, não qualidade.
- **Mistura:** convenção de SNR (ruído escalado contra alvo + interferente) e alvo seco não alinhado à RIR aceitos no
  M0, a conferir contra a spec §4.4.
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


def main() -> None:
    i2 = i2_own()
    runs = {n: _load(n) for n in ("env", "i1", "i2_clearcore_fixture", "film_asset", "gating_parity",
                                  "offline_vs_runtime", "ecapa_tract", "disk", "loader", "step_b16", "step_b32",
                                  "step_b16_teacher", "beta_rule")}
    gates, budget = load_gates(), load_budget()
    minimal = yaml.safe_load((PATHS.repo / "configs/data/minimal-set-v1.yaml").read_text())
    proj = project_disk(minimal, runs["disk"])
    already = sum(v or 0 for k, v in runs["disk"]["dirs_bytes"].items() if k in ("venv", "tract_check_target",
                                                                                  "rustup", "data")) / 1e9
    free_gb = shutil.disk_usage(PATHS.data).free / 1e9 + already
    verdicts = go_no_go(budget, proj["total_gb"], free_gb, runs["step_b16_teacher"], runs["loader"])
    resampler = json.loads((V1_DIR / "manifest.json").read_text())["coefficients"]
    sections = [
        ("Resumo", "\n".join(f"- {r['item']}: **{r['verdict']}**" for r in verdicts)
         + f"\n- I1: **{'aprovado' if runs['i1'].get('pass') else 'REPROVADO'}**; "
           f"I2 (asset do repo): **{'aprovado' if i2.get('pass') else 'REPROVADO'}**\n"
           f"- B fixado: {runs['beta_rule'].get('beta_bound')}\n"
           f"- Réplica do gating: {'paridade ok' if runs['gating_parity'].get('pass') else 'FALHOU'}\n"
           f"- ECAPA no tract: { {m: v.get('all_pass') for m, v in runs['ecapa_tract'].items()} }"),
        ("Decisões do orquestrador", DECISOES),
        ("Ambiente", fmt(runs["env"])),
        ("Identidade (I1, I2)", fmt({"i1": runs["i1"], "i2_fixture_clearcore": runs["i2_clearcore_fixture"],
                                     "i2_asset_do_repo": i2, "film_asset": runs["film_asset"]})),
        ("Congelamentos", fmt({"gates_v1_sha256": gates.sha256, "budget_v1_sha256": budget.sha256,
                               "resampler_v1": resampler})),
        ("Réplica do gating × tract-check", fmt(runs["gating_parity"])),
        ("Emulação offline × runtime (dado para o M2)", fmt(runs["offline_vs_runtime"])),
        ("Protótipo fbank + ECAPA no tract", fmt({m: {"all_pass": v.get("all_pass"),
                                                     "falhas": [c for c in v.get("cases", []) if not c["passed"]]}
                                                 for m, v in runs["ecapa_tract"].items()})),
        ("Disco (fatores medidos e projeção do conjunto mínimo)", fmt({"medido": runs["disk"], "projecao": proj,
                                                                       "livre_mais_ja_usado_gb": free_gb})),
        ("Loader", fmt(runs["loader"])),
        ("Passo de treino na GPU", fmt({k: runs[k] for k in ("step_b16", "step_b32", "step_b16_teacher")})),
        ("Regra de B", fmt(runs["beta_rule"])),
        ("Pendências", "- CV 27.0 pt: " + ("manifest `cv27-pt.jsonl` presente"
                                            if (PATHS.data / "manifests/cv27-pt.jsonl").is_file()
                                            else "não ingerido no piloto (custo medido pelos fatores por SR)")
         + "\n- Disco: fatores FLAC de fala por fonte (VCTK a 48 kHz, LibriSpeech dev-clean a 16 kHz, demais SR "
           "interpolados); ruído pelo agregado por SR; `tract_check_target` medido no checkout principal."
         + "\n- Decisões de planejamento 1-15 do plano bootstrap+M0 valem como registradas lá."),
        ("Próximos passos", PROXIMOS),
    ]
    md = render_markdown({"title": "Clearcore Train — relatório M0 (piloto)", "sections": sections,
                          "go_no_go": verdicts})
    (PATHS.repo / "docs").mkdir(exist_ok=True)
    (PATHS.repo / "docs/m0-report.md").write_text(md)
    (RUNS / "report.json").write_text(json.dumps({"go_no_go": verdicts, "disk_projection": proj, "i2_own": i2},
                                                 indent=2, default=str) + "\n")
    print(md)


if __name__ == "__main__":
    main()
