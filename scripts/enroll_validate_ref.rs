// Referência da validação de enrollment: cópia de constantes, `mean_square_dbfs` e `validate_enrollment_audio` de
// clearcore/crates/model/src/enrollment.rs (MIT OR Apache-2.0), reduzida a std. Só serve para gerar o golden de
// tests/golden/enroll_validation_v1.json (scripts/gen_enroll_validation_golden.py); não entra em nenhum build.
use std::{env, fs};

const ENROLLMENT_MIN_SAMPLES: usize = 96_000;
const ENROLLMENT_MAX_SAMPLES: usize = 192_000;
const ENROLLMENT_MAX_PEAK: f32 = 0.99;
const ENROLLMENT_MIN_RMS_DBFS: f32 = -40.0;
const ENROLLMENT_MIN_ACTIVE_FRACTION: f32 = 0.60;
const VAD_FRAME_SAMPLES: usize = 320;
const VAD_ABSOLUTE_FLOOR_DBFS: f32 = -50.0;
const VAD_RELATIVE_RANGE_DB: f32 = 30.0;

#[allow(clippy::cast_precision_loss)]
const fn count_to_f32(value: usize) -> f32 {
    value as f32
}

fn mean_square_dbfs(samples: &[f32]) -> f32 {
    let sum: f64 = samples.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
    let mean = sum / samples.len().max(1) as f64;
    let db = (10.0 * (mean + 1e-12).log10()) as f32;
    db
}

// (status, valor do erro ou -1, peak, rms_dbfs, active_fraction) -- os três últimos só em "ok".
fn validate(samples: &[f32]) -> (&'static str, f64, f32, f32, f32) {
    if samples.len() < ENROLLMENT_MIN_SAMPLES {
        return ("too_short", samples.len() as f64, 0.0, 0.0, 0.0);
    }
    if samples.len() > ENROLLMENT_MAX_SAMPLES {
        return ("too_long", samples.len() as f64, 0.0, 0.0, 0.0);
    }
    if let Some(index) = samples.iter().position(|s| !s.is_finite()) {
        return ("non_finite", index as f64, 0.0, 0.0, 0.0);
    }
    let peak = samples.iter().fold(0.0_f32, |acc, s| acc.max(s.abs()));
    if peak > ENROLLMENT_MAX_PEAK {
        return ("clipping", f64::from(peak), 0.0, 0.0, 0.0);
    }
    let rms_dbfs = mean_square_dbfs(samples);
    if rms_dbfs <= ENROLLMENT_MIN_RMS_DBFS {
        return ("too_quiet", f64::from(rms_dbfs), 0.0, 0.0, 0.0);
    }
    let frame_levels: Vec<f32> = samples.chunks_exact(VAD_FRAME_SAMPLES).map(mean_square_dbfs).collect();
    let loudest = frame_levels.iter().fold(f32::MIN, |acc, level| acc.max(*level));
    let threshold = VAD_ABSOLUTE_FLOOR_DBFS.max(loudest - VAD_RELATIVE_RANGE_DB);
    let active = frame_levels.iter().filter(|level| **level >= threshold).count();
    let active_speech_fraction = count_to_f32(active) / count_to_f32(frame_levels.len());
    if active_speech_fraction < ENROLLMENT_MIN_ACTIVE_FRACTION {
        return ("insufficient_speech", f64::from(active_speech_fraction), 0.0, 0.0, 0.0);
    }
    ("ok", -1.0, peak, rms_dbfs, active_speech_fraction)
}

fn main() {
    // uso: enroll_validate_ref <caso.f32>...  -> uma linha JSON por arquivo
    for path in env::args().skip(1) {
        let bytes = fs::read(&path).expect("ler caso");
        let samples: Vec<f32> =
            bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let (status, value, peak, rms, frac) = validate(&samples);
        let name = std::path::Path::new(&path).file_stem().unwrap().to_string_lossy().to_string();
        println!(
            "{{\"case\":\"{name}\",\"status\":\"{status}\",\"value\":{value:?},\"peak\":{peak:?},\"rms_dbfs\":{rms:?},\"active_fraction\":{frac:?}}}"
        );
    }
}
