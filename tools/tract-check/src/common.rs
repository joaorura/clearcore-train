//! Utilidades compartilhadas: sinal sintético do `parity_film.rs` do Clearcore, E/S f32 e carga do DfTract.
use std::{fs, path::Path};

use anyhow::{bail, Context, Result};
use deep_filter::tract::{DfParams, DfTract, RuntimeParams};
use ndarray::{Array2, ArrayView2};
use sha2::{Digest, Sha256};

pub const HOP: usize = 480;

/// Sinal idêntico ao de `crates/model/tests/parity_film.rs` (pilha harmônica de 140 Hz, envelope de
/// 3 Hz, ruído LCG). Precisa reproduzir a aritmética bit a bit: sem `mul_add`.
#[allow(clippy::suboptimal_flops, clippy::cast_possible_truncation)]
pub fn synthetic(frames: usize) -> Vec<f32> {
    let mut state: u32 = 0x1234_5678;
    (0..frames * HOP)
        .map(|index| {
            let t = f64::from(u32::try_from(index).unwrap_or(u32::MAX)) / 48_000.0;
            let envelope = 0.5 * (1.0 + (2.0 * std::f64::consts::PI * 3.0 * t).sin());
            let voiced: f64 = (1..=12)
                .map(|h| (2.0 * std::f64::consts::PI * 140.0 * f64::from(h) * t).sin() / f64::from(h))
                .sum();
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = f64::from(state >> 8) / 8_388_608.0 - 1.0;
            (0.08 * envelope * voiced + 0.02 * noise).clamp(-0.99, 0.99) as f32
        })
        .collect()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn to_le_bytes(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

pub fn read_f32_le(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("lendo {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!("{}: tamanho {} não é múltiplo de 4", path.display(), bytes.len());
    }
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

pub fn write_f32_le(path: &Path, data: &[f32]) -> Result<()> {
    fs::write(path, to_le_bytes(data)).with_context(|| format!("gravando {}", path.display()))
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        bail!("comprimentos diferentes: {} e {}", a.len(), b.len());
    }
    Ok(a.iter().zip(b).fold(0.0_f32, |acc, (x, y)| acc.max((x - y).abs())))
}

pub fn load_model(asset: &Path, params: &RuntimeParams) -> Result<DfTract> {
    let dfp = DfParams::new(asset.to_path_buf())?;
    DfTract::new(dfp, params)
}

pub fn process_frame(model: &mut DfTract, frame: &[f32]) -> Result<(Vec<f32>, f32)> {
    let input = ArrayView2::from_shape((1, HOP), frame)?;
    let mut output = Array2::<f32>::zeros((1, HOP));
    let lsnr = model.process(input, output.view_mut())?;
    Ok((output.into_raw_vec(), lsnr))
}

pub fn run_signal(model: &mut DfTract, signal: &[f32]) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(signal.len());
    for frame in signal.chunks_exact(HOP) {
        out.extend(process_frame(model, frame)?.0);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn synthetic_is_bounded_and_deterministic() {
        let a = synthetic(10);
        assert_eq!(a.len(), 10 * HOP);
        assert!(a.iter().all(|x| x.abs() <= 0.99));
        assert_eq!(to_le_bytes(&a), to_le_bytes(&synthetic(10)));
    }

    #[test]
    fn max_abs_diff_rejects_length_mismatch() {
        assert!(max_abs_diff(&[0.0], &[0.0, 1.0]).is_err());
        assert_eq!(max_abs_diff(&[1.0, -2.0], &[1.5, -2.0]).unwrap(), 0.5);
    }
}
