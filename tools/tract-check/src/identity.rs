//! I2 (spec §6.3): asset com FiLM em identidade × golden de 1.000 quadros do upstream, max_abs_diff == 0.
use std::{fs, path::Path};

use anyhow::{bail, Result};
use deep_filter::tract::{FilmVectors, RuntimeParams};
use serde_json::json;

use crate::common::{load_model, max_abs_diff, read_f32_le, run_signal, sha256_hex, synthetic, to_le_bytes, HOP};

pub const UPSTREAM_GOLDEN_SHA256: &str = "c041ee1fa4b2ce4bdad4f07e28b8fdcb901c56bab20380d17abd254d5c9736aa";

fn identity_vectors(hidden: usize) -> FilmVectors {
    FilmVectors {
        gamma_enc: vec![1.0; hidden],
        beta_enc: vec![0.0; hidden],
        gamma_df: vec![1.0; hidden],
        beta_df: vec![0.0; hidden],
    }
}

pub fn run(args: &[String]) -> Result<bool> {
    let [film_asset, approved_asset, golden_path, out_json] = args else {
        bail!("uso: tract-check identity <film_asset.tar.gz> <approved_asset.bin> <golden.f32> <out.json>");
    };
    let golden_bytes = fs::read(golden_path)?;
    let golden_sha = sha256_hex(&golden_bytes);
    if golden_sha != UPSTREAM_GOLDEN_SHA256 {
        bail!("golden {golden_path}: sha256 {golden_sha} != {UPSTREAM_GOLDEN_SHA256}");
    }
    let golden = read_f32_le(Path::new(golden_path))?;
    let frames = golden.len() / HOP;
    let signal = synthetic(frames);
    let params = RuntimeParams::default();

    let approved = run_signal(&mut load_model(Path::new(approved_asset), &params)?, &signal)?;
    let mut film_default = load_model(Path::new(film_asset), &params)?;
    let film_hidden = film_default.film_hidden();
    let out_default = run_signal(&mut film_default, &signal)?;
    let mut film_explicit = load_model(Path::new(film_asset), &params)?;
    film_explicit.set_film(&identity_vectors(film_hidden.unwrap_or(256)))?;
    let out_explicit = run_signal(&mut film_explicit, &signal)?;

    let sha_default = sha256_hex(&to_le_bytes(&out_default));
    let sha_explicit = sha256_hex(&to_le_bytes(&out_explicit));
    let sha_approved = sha256_hex(&to_le_bytes(&approved));
    let pass = film_hidden == Some(256)
        && sha_default == UPSTREAM_GOLDEN_SHA256
        && sha_explicit == UPSTREAM_GOLDEN_SHA256;
    let report = json!({
        "frames": frames,
        "film_hidden": film_hidden,
        "golden_sha256": golden_sha,
        "approved_out_sha256": sha_approved,
        "film_default_out_sha256": sha_default,
        "film_explicit_out_sha256": sha_explicit,
        "max_abs_diff_default_vs_golden": max_abs_diff(&out_default, &golden)?,
        "max_abs_diff_explicit_vs_golden": max_abs_diff(&out_explicit, &golden)?,
        "max_abs_diff_explicit_vs_approved": max_abs_diff(&out_explicit, &approved)?,
        "approved_bit_exact_vs_golden": sha_approved == UPSTREAM_GOLDEN_SHA256,
        "pass": pass,
    });
    fs::write(out_json, serde_json::to_string_pretty(&report)? + "\n")?;
    println!("{report}");
    Ok(pass)
}
