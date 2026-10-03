//! Caminho do runtime com gating por lsnr: `DfTract::process` quadro a quadro (spec §3.5).
use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{bail, Context, Result};
use deep_filter::tract::{FilmVectors, RuntimeParams};
use serde_json::json;

use crate::common::{load_model, process_frame, read_f32_le, sha256_hex, to_le_bytes, write_f32_le, HOP};

pub const STAGE_ZERO: u8 = 0;
pub const STAGE_ERB_ONLY: u8 = 1;
pub const STAGE_ERB_DF: u8 = 2;
pub const STAGE_NONE: u8 = 3;
pub const STAGE_SILENT: u8 = 4;

pub fn stage_code(stages: (bool, bool, bool)) -> u8 {
    match stages {
        (false, true, false) => STAGE_ZERO,
        (true, false, false) => STAGE_ERB_ONLY,
        (true, false, true) => STAGE_ERB_DF,
        (false, false, false) => STAGE_NONE,
        _ => u8::MAX,
    }
}

/// Mesma conta de `DfTract::process_with_spec_hook`: soma sequencial de x² em f32, dividida pelo comprimento.
#[allow(clippy::cast_precision_loss)]
pub fn mean_square(frame: &[f32]) -> f32 {
    frame.iter().fold(0f32, |acc, x| acc + x.powi(2)) / frame.len() as f32
}

fn read_film(path: &Path, hidden: usize) -> Result<FilmVectors> {
    let v = read_f32_le(path)?;
    if v.len() != 4 * hidden {
        bail!("{}: esperados {} valores f32, lidos {}", path.display(), 4 * hidden, v.len());
    }
    Ok(FilmVectors {
        gamma_enc: v[..hidden].to_vec(),
        beta_enc: v[hidden..2 * hidden].to_vec(),
        gamma_df: v[2 * hidden..3 * hidden].to_vec(),
        beta_df: v[3 * hidden..].to_vec(),
    })
}

fn stage_name(code: u8) -> &'static str {
    match code {
        STAGE_ZERO => "zero",
        STAGE_ERB_ONLY => "erb_only",
        STAGE_ERB_DF => "erb_df",
        STAGE_NONE => "none",
        STAGE_SILENT => "silent",
        _ => "invalid",
    }
}

#[allow(clippy::cast_precision_loss)]
pub fn run(args: &[String]) -> Result<bool> {
    let (asset, input, prefix, film) = match args {
        [a, i, p] => (a, i, p, None),
        [a, i, p, f] => (a, i, p, Some(f)),
        _ => bail!("uso: tract-check gating <asset.tar.gz> <in_48k.f32> <out_prefix> [film.f32]"),
    };
    let signal = read_f32_le(Path::new(input))?;
    let frames = signal.len() / HOP;
    if frames == 0 {
        bail!("entrada com menos de um hop");
    }
    let mut model = load_model(Path::new(asset), &RuntimeParams::default())?;
    if let Some(f) = film {
        let hidden = model.film_hidden().context("asset sem entradas FiLM")?;
        model.set_film(&read_film(Path::new(f), hidden)?)?;
    }
    let mut out = Vec::with_capacity(frames * HOP);
    let mut lsnr = Vec::with_capacity(frames);
    let mut stages = Vec::with_capacity(frames);
    for frame in signal.chunks_exact(HOP) {
        let silent = mean_square(frame) < 1e-7;
        let (y, l) = process_frame(&mut model, frame)?;
        stages.push(if silent { STAGE_SILENT } else { stage_code(model.apply_stages(l)) });
        lsnr.push(l);
        out.extend(y);
    }
    write_f32_le(Path::new(&format!("{prefix}.out.f32")), &out)?;
    write_f32_le(Path::new(&format!("{prefix}.lsnr.f32")), &lsnr)?;
    fs::write(format!("{prefix}.stage.u8"), &stages)?;
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for s in &stages {
        *counts.entry(stage_name(*s)).or_default() += 1;
    }
    let fractions: BTreeMap<&str, f64> =
        counts.iter().map(|(k, v)| (*k, *v as f64 / frames as f64)).collect();
    let report = json!({
        "frames": frames,
        "fractions": fractions,
        "out_sha256": sha256_hex(&to_le_bytes(&out)),
        "film": film.is_some(),
    });
    fs::write(format!("{prefix}.json"), serde_json::to_string_pretty(&report)? + "\n")?;
    println!("{report}");
    Ok(!counts.contains_key("invalid"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_codes_follow_apply_stages() {
        assert_eq!(stage_code((false, true, false)), STAGE_ZERO);
        assert_eq!(stage_code((true, false, false)), STAGE_ERB_ONLY);
        assert_eq!(stage_code((true, false, true)), STAGE_ERB_DF);
        assert_eq!(stage_code((false, false, false)), STAGE_NONE);
        assert_eq!(stage_code((true, true, true)), u8::MAX);
    }

    #[test]
    fn mean_square_matches_definition() {
        assert_eq!(mean_square(&[0.0; 480]), 0.0);
        assert_eq!(mean_square(&[2.0, 0.0]), 2.0);
    }
}
