"""Config congelada do estágio F e regra de seleção de checkpoint (M2 Tarefa 1).

`load_stage_f_config` e `load_selection` conferem o SHA-256 do arquivo (`*.sha256` ao lado) e validam contra a spec rev. 2:
o que sai do pré-registrado é recusado com `M2ConfigError`. Os valores vêm de `configs/train/stage-f-v1.yaml` e
`configs/eval/m2-selection-v1.yaml`; os dois são congelados antes da Tarefa 6 (T3).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from cctrain.eval.gates import verify_frozen
from cctrain.paths import repo_root

STAGE_F_V1 = repo_root() / "configs" / "train" / "stage-f-v1.yaml"
SELECTION_V1 = repo_root() / "configs" / "eval" / "m2-selection-v1.yaml"
_GAMMA_BOUND_MAX = math.log(100.0)  # spec §3.2: [e^-Γ, e^Γ] ⊂ [0,001; 100]
_BETA_MIN, _BETA_MAX = 4.0, 50.0


class M2ConfigError(ValueError):
    """Config do M2 fora da spec ou com chaves inesperadas."""


def _exact(d: object, keys: set[str], where: str) -> Mapping:
    if not isinstance(d, Mapping) or set(d) != keys:
        got = sorted(d) if isinstance(d, Mapping) else type(d).__name__
        raise M2ConfigError(f"{where}: chaves esperadas {sorted(keys)}, recebidas {got}")
    return d


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise M2ConfigError(msg)


# ------------------------------------------------------------------------------------------------ estágio F
@dataclass(frozen=True)
class InitCfg:
    generator_checkpoint: str
    generator_checkpoint_sha256: str
    base: str
    base_state_sha256: str


@dataclass(frozen=True)
class FreezeCfg:
    encoder: bool
    batchnorm: str
    base_trainable: bool


@dataclass(frozen=True)
class OptimizerCfg:
    name: str
    lr: float
    schedule: str
    epochs: int
    warmup_steps: int
    weight_decay: float
    grad_clip: float


@dataclass(frozen=True)
class DataCfg:
    batch_size: int
    segment_s: float
    epoch_audio_hours: float
    steps_per_epoch: int


@dataclass(frozen=True)
class BaseLossCfg:
    factor: float
    factor_complex: float
    gamma: float
    fft_sizes: tuple[int, ...]


@dataclass(frozen=True)
class LossCfg:
    base: BaseLossCfg
    lambda_t: float
    lambda_os: float
    os_average: str
    lambda_gate: float
    ignore_above_max_freq: bool


@dataclass(frozen=True)
class FilmCfg:
    log_gamma_bound: float
    beta_bound: float
    p_drop: float


@dataclass(frozen=True)
class BandQuota:
    full_band_min: float
    khz24_target: float
    khz16_max: float


@dataclass(frozen=True)
class MixingCfg:
    s: float
    base_fractions: Mapping[str, float]
    mask0_target: str
    snr_grid_db: tuple[float, ...]
    sir_db: tuple[float, float]
    n_interferers: tuple[int, ...]
    p_two_interferers: float
    p_rir: float
    segment_s: float
    band_quota: BandQuota

    def fractions(self) -> dict[str, float]:
        """Fração de cada tipo de mistura (spec §4.4): 35/20/35 ÷ 0,9 · (1−s), mais `s` sem alvo."""
        k = (1.0 - self.s) / 0.9
        out = {n: f * k for n, f in self.base_fractions.items()}
        out["no_target"] = self.s
        return out


@dataclass(frozen=True)
class EnrollmentCfg:
    p_enroll_denoise: float
    bank: str
    duration_s: tuple[float, float]
    never_mixture_utterance: bool


@dataclass(frozen=True)
class CheckpointCfg:
    every_min: int
    keep_last: int
    keep_best: bool
    pin_epoch: int


@dataclass(frozen=True)
class Ab1Cfg:
    mode: str
    arms_p_enroll_denoise: tuple[float, ...]
    seeds_per_arm: int
    inherited_branches: bool
    label: str


@dataclass(frozen=True)
class Ab2Cfg:
    mode: str
    p_drop: tuple[float, ...]
    s: tuple[float, ...]


@dataclass(frozen=True)
class AblationsCfg:
    ab1: Ab1Cfg
    ab2: Ab2Cfg


@dataclass(frozen=True)
class StageFConfig:
    version: int
    stage: str
    seed: int
    init: InitCfg
    freeze: FreezeCfg
    optimizer: OptimizerCfg
    data: DataCfg
    loss: LossCfg
    film: FilmCfg
    mixing: MixingCfg
    enrollment: EnrollmentCfg
    checkpoint: CheckpointCfg
    ablations: AblationsCfg
    sha256: str = ""

    def config_hash(self) -> str:
        d = dataclasses.asdict(self)
        d.pop("sha256")
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()

    def validate(self) -> StageFConfig:
        o, d, lo, f, m, e, c, a = (self.optimizer, self.data, self.loss, self.film, self.mixing, self.enrollment,
                                   self.checkpoint, self.ablations)
        _need(self.stage == "F", "stage deve ser F")
        _need(self.freeze.encoder, "o encoder fica congelado no estágio F")
        _need(self.freeze.batchnorm == "eval_frozen", "BatchNorm em eval() com parâmetros congelados (spec §5.3)")
        _need(self.freeze.base_trainable, "no estágio F a base treina")
        _need(o.name == "adamw" and o.schedule == "cosine", "otimizador AdamW com cosseno")
        _need(o.lr == 1e-4, f"lr {o.lr} != 1e-4 (spec §5.3)")
        _need(o.epochs == 20, f"épocas {o.epochs} != 20 (orçamento: go_hours_max do budget-v1)")
        _need(o.warmup_steps == 1000, "warmup de 1.000 passos")
        _need(d.batch_size == 16 and d.segment_s == 3.0, f"batch {d.batch_size}/segmento {d.segment_s} != 16/3 s")
        _need(d.steps_per_epoch == 7500, "uma época = 7.500 passos (100 h virtuais, budget-v1)")
        _need(lo.lambda_t == 500.0 and lo.lambda_os == 500.0, "λ_T e λ_OS = 500 (spec §5.4)")
        _need(lo.lambda_t > 0 and lo.lambda_os > 0 and lo.lambda_gate > 0, "λ_T, λ_OS e λ_g positivos")
        _need(lo.lambda_gate == 1e-3, "λ_g = 1e-3")
        _need(lo.os_average == "time_frequency", "L_OS é a média tempo-frequência")
        _need(0.0 < f.log_gamma_bound <= _GAMMA_BOUND_MAX, f"Γ={f.log_gamma_bound} fora de (0, ln 100]")
        _need(math.isclose(f.log_gamma_bound, math.log(10.0)), "Γ do estágio F = ln 10")
        _need(_BETA_MIN <= f.beta_bound <= _BETA_MAX, f"B={f.beta_bound} fora de [{_BETA_MIN}, {_BETA_MAX}]")
        _need(0.0 <= f.p_drop < 1.0, f"p_drop {f.p_drop} fora de [0, 1)")
        _need(0.0 <= m.s <= 0.10, f"s={m.s} fora de [0, 10%] (AB2 varre {{0; 5; 10}}%)")
        _need(math.isclose(sum(m.base_fractions.values()), 0.9) and set(m.base_fractions) == {
            "target_noise", "target_interferer", "target_interferer_noise"}, "frações base §4.4 somam 0,9")
        _need(m.snr_grid_db == (-5.0, 0.0, 5.0, 10.0, 20.0, 40.0), "grade de SNR do upstream (§4.4)")
        _need(m.sir_db == (-5.0, 25.0), "SIR U[−5, 25] dB")
        _need(set(m.n_interferers) <= {1, 2} and 0.0 <= m.p_two_interferers <= 1.0, "1 ou 2 interferentes")
        _need(m.p_rir == 0.3 and m.segment_s == 3.0, "RIR p=0,3 e trechos de 3 s")
        q = m.band_quota
        _need(q.full_band_min >= 0.5 and 0.0 <= q.khz16_max <= 0.2
              and math.isclose(q.full_band_min + q.khz24_target + q.khz16_max, 1.0),
              f"cota de banda fora da spec §4.3 (≥ 50% full-band, ~30% 24 kHz, ≤ 20% 16 kHz): {q}")
        _need(0.0 <= e.p_enroll_denoise <= 1.0, f"p_enroll_denoise {e.p_enroll_denoise} fora de [0, 1]")
        _need(e.duration_s == (6.0, 12.0) and e.never_mixture_utterance, "enrollment 6-12 s, nunca a fala da mistura")
        _need(c.every_min == 30 and c.keep_last == 3 and c.keep_best, "checkpoint: 30 min, 3 últimos + o melhor")
        _need(0 < c.pin_epoch < o.epochs and c.pin_epoch == round(0.7 * o.epochs), "época fixada = 70% do cosseno")
        b = a.ab1
        _need(b.mode == "full_run" and not b.inherited_branches,
              "AB1: rodada completa, sem ramos herdados (decisão do dono, 2026-10-03)")
        _need(b.seeds_per_arm == 1, "AB1: 1 seed por braço (decisão do dono, 2026-10-03)")
        _need(b.arms_p_enroll_denoise == (0.0, 0.5), "AB1: braços p_enroll_denoise {0; 0,5}")
        _need(a.ab2.p_drop == (0.1, 0.3) and a.ab2.s == (0.0, 0.10), "AB2: p_drop {0,1; 0,3}, s {0; 10}%")
        return self


def _t(x) -> tuple:
    return tuple(x)


def parse_stage_f_config(raw: Mapping, sha256: str = "") -> StageFConfig:
    top = {"version", "stage", "seed", "init", "freeze", "optimizer", "data", "loss", "film", "mixing", "enrollment",
           "checkpoint", "ablations"}
    r = _exact(raw, top, "stage-f")
    loss = _exact(r["loss"], {"base", "lambda_t", "lambda_os", "os_average", "lambda_gate", "ignore_above_max_freq"},
                  "loss")
    base = _exact(loss["base"], {"factor", "factor_complex", "gamma", "fft_sizes"}, "loss.base")
    mix = _exact(r["mixing"], {"s", "base_fractions", "mask0_target", "snr_grid_db", "sir_db", "n_interferers",
                               "p_two_interferers", "p_rir", "segment_s", "band_quota"}, "mixing")
    quota = _exact(mix["band_quota"], {"full_band_min", "khz24_target", "khz16_max"}, "mixing.band_quota")
    enr = _exact(r["enrollment"], {"p_enroll_denoise", "bank", "duration_s", "never_mixture_utterance"}, "enrollment")
    abl = _exact(r["ablations"], {"ab1", "ab2"}, "ablations")
    ab1 = _exact(abl["ab1"], {"mode", "arms_p_enroll_denoise", "seeds_per_arm", "inherited_branches", "label"},
                 "ablations.ab1")
    ab2 = _exact(abl["ab2"], {"mode", "p_drop", "s"}, "ablations.ab2")
    try:
        cfg = StageFConfig(
            version=int(r["version"]), stage=r["stage"], seed=int(r["seed"]),
            init=InitCfg(**_exact(r["init"], {"generator_checkpoint", "generator_checkpoint_sha256", "base",
                                              "base_state_sha256"}, "init")),
            freeze=FreezeCfg(**_exact(r["freeze"], {"encoder", "batchnorm", "base_trainable"}, "freeze")),
            optimizer=OptimizerCfg(**_exact(r["optimizer"], {"name", "lr", "schedule", "epochs", "warmup_steps",
                                                             "weight_decay", "grad_clip"}, "optimizer")),
            data=DataCfg(**_exact(r["data"], {"batch_size", "segment_s", "epoch_audio_hours", "steps_per_epoch"},
                                  "data")),
            loss=LossCfg(base=BaseLossCfg(**{**base, "fft_sizes": _t(base["fft_sizes"])}),
                         **{k: v for k, v in loss.items() if k != "base"}),
            film=FilmCfg(**_exact(r["film"], {"log_gamma_bound", "beta_bound", "p_drop"}, "film")),
            mixing=MixingCfg(**{**{k: v for k, v in mix.items() if k != "band_quota"},
                                "snr_grid_db": _t(float(x) for x in mix["snr_grid_db"]),
                                "sir_db": _t(float(x) for x in mix["sir_db"]),
                                "n_interferers": _t(mix["n_interferers"]), "band_quota": BandQuota(**quota)}),
            enrollment=EnrollmentCfg(**{**enr, "duration_s": _t(float(x) for x in enr["duration_s"])}),
            checkpoint=CheckpointCfg(**_exact(r["checkpoint"], {"every_min", "keep_last", "keep_best", "pin_epoch"},
                                              "checkpoint")),
            ablations=AblationsCfg(
                ab1=Ab1Cfg(**{**ab1, "arms_p_enroll_denoise": _t(float(x) for x in ab1["arms_p_enroll_denoise"])}),
                ab2=Ab2Cfg(mode=ab2["mode"], p_drop=_t(float(x) for x in ab2["p_drop"]),
                           s=_t(float(x) for x in ab2["s"]))),
            sha256=sha256)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, M2ConfigError):
            raise
        raise M2ConfigError(f"stage-f: valor inválido ({exc})") from exc
    return cfg.validate()


def load_stage_f_config(path: Path = STAGE_F_V1) -> StageFConfig:
    path = Path(path)
    digest = verify_frozen(path)
    return parse_stage_f_config(yaml.safe_load(path.read_text()), sha256=digest)


# ------------------------------------------------------------------------------------------------ seleção
@dataclass(frozen=True)
class EvaluationCfg:
    split: str
    path: str
    every: str


@dataclass(frozen=True)
class EligibilityCfg:
    condition: str
    reading: str
    delta_si_sdr_db_min: float
    delta_stoi_min: float


@dataclass(frozen=True)
class ScoreCfg:
    metric: str
    condition: str
    sir_db: float
    aggregation: str
    speakers: str


@dataclass(frozen=True)
class TieCfg:
    tolerance_db: float
    prefer: str


@dataclass(frozen=True)
class EarlyStoppingCfg:
    from_epoch: int
    patience_evaluations: int
    min_improvement_db: float


@dataclass(frozen=True)
class OnFailCfg:
    gates: tuple[str, ...]
    action: str
    export: bool
    gates_v3: bool


@dataclass(frozen=True)
class SelectionRule:
    version: int
    evaluation: EvaluationCfg
    eligibility: EligibilityCfg
    score: ScoreCfg
    tie: TieCfg
    early_stopping: EarlyStoppingCfg
    no_eligible_action: str
    forbidden_inputs: tuple[str, ...]
    g1_dev_reading: str
    on_stage_f_fail: OnFailCfg
    ab1_design: str
    speaker_sex_source: str
    sha256: str = ""

    def validate(self) -> SelectionRule:
        ev, el, sc, ti, es = self.evaluation, self.eligibility, self.score, self.tie, self.early_stopping
        _need(ev.split == "dev-select" and ev.path == "offline_gating_emulated" and ev.every == "epoch",
              "avaliação a cada época no dev-select, caminho offline com gating emulado")
        _need(el.condition == "C0" and el.reading == "point_estimate", "elegibilidade: C0, estimativa pontual")
        _need(el.delta_si_sdr_db_min == -0.5 and el.delta_stoi_min == -0.003, "margens de elegibilidade: −0,5 dB / −0,003")
        _need(sc.metric == "delta_si_sdr_db" and sc.condition == "C3" and sc.sir_db == 0.0
              and sc.aggregation == "mean_per_speaker" and sc.speakers == "all_dev_select", "escore: ΔSI-SDR médio C3, SIR 0")
        _need(ti.tolerance_db == 0.05 and ti.prefer == "oldest", "empate < 0,05 dB: o mais antigo")
        _need((es.from_epoch, es.patience_evaluations, es.min_improvement_db) == (8, 4, 0.1),
              "early stopping: época 8, 4 avaliações, 0,1 dB")
        _need(self.no_eligible_action == "no_go_t3", "nenhum elegível: NO-GO do T3")
        _need({"dev-gate", "test"} <= set(self.forbidden_inputs), "dev-gate e teste não entram em nenhuma escolha")
        _need(self.g1_dev_reading == "ci95_bound_toward_failure",
              "G1 no dev lido pelo limite do IC do lado da reprovação (decisão do dono, 2026-10-03)")
        f = self.on_stage_f_fail
        _need(f.gates == ("G1", "G2", "G4") and f.action == "stop_and_diagnose" and f.export is False
              and f.gates_v3 is False, "F reprova G1/G2/G4: parar e diagnosticar, sem export e sem gates-v3")
        _need(self.ab1_design == "full_run_one_seed_per_arm", "AB1: 1 seed completa por braço")
        _need(self.speaker_sex_source == "declared_metadata_only", "sexo só de metadado declarado")
        return self


def load_selection(path: Path = SELECTION_V1) -> SelectionRule:
    path = Path(path)
    digest = verify_frozen(path)
    r = _exact(yaml.safe_load(path.read_text()), {
        "version", "gates_file", "evaluation", "eligibility", "score", "choose", "tie", "early_stopping",
        "no_eligible_action", "forbidden_inputs", "g1_dev_reading", "on_stage_f_fail", "ab1",
        "speaker_sex_source"}, "m2-selection")
    try:
        _need(r["choose"] == "highest_score_among_eligible", "escolhe o elegível de maior escore")
        fail = _exact(r["on_stage_f_fail"], {"gates", "action", "export", "gates_v3"}, "on_stage_f_fail")
        rule = SelectionRule(
            version=int(r["version"]),
            evaluation=EvaluationCfg(**_exact(r["evaluation"], {"split", "path", "every"}, "evaluation")),
            eligibility=EligibilityCfg(**_exact(r["eligibility"], {"condition", "reading", "delta_si_sdr_db_min",
                                                                   "delta_stoi_min"}, "eligibility")),
            score=ScoreCfg(**_exact(r["score"], {"metric", "condition", "sir_db", "aggregation", "speakers"}, "score")),
            tie=TieCfg(**_exact(r["tie"], {"tolerance_db", "prefer"}, "tie")),
            early_stopping=EarlyStoppingCfg(**_exact(r["early_stopping"], {"from_epoch", "patience_evaluations",
                                                                           "min_improvement_db"}, "early_stopping")),
            no_eligible_action=r["no_eligible_action"], forbidden_inputs=_t(r["forbidden_inputs"]),
            g1_dev_reading=r["g1_dev_reading"], on_stage_f_fail=OnFailCfg(**{**fail, "gates": _t(fail["gates"])}),
            ab1_design=_exact(r["ab1"], {"design", "label"}, "ab1")["design"],
            speaker_sex_source=r["speaker_sex_source"], sha256=digest)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, M2ConfigError):
            raise
        raise M2ConfigError(f"m2-selection: valor inválido ({exc})") from exc
    return rule.validate()
