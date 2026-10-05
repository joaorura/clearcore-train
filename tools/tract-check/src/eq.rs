//! Carga do neural_eq.onnx no tract 0.19.16: entrada feat fixada em f32 [batch, 32, 3].
use std::{fs, io::Cursor, path::Path};

use anyhow::{bail, Result};
use serde_json::json;
use tract_onnx::prelude::*;

use crate::common::read_f32_le;
fn run_eq(bytes: &[u8], feat: &[f32], batch: usize) -> TractResult<Vec<f32>> {
    let mut model = onnx().model_for_read(&mut Cursor::new(bytes))?;
    model.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(batch, 32, 3)))?;
    model.set_output_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(batch, 32)))?;
    let runnable = model.into_optimized()?.into_runnable()?;
    let input = Tensor::from_shape(&[batch, 32, 3], feat)?;
    let outputs = runnable.run(tvec!(input.into()))?;
    let gains = outputs[0].as_slice::<f32>()?.to_vec();
    Ok(gains)
}

pub fn run(args: &[String]) -> Result<bool> {
    if args.len() < 3 {
        bail!("uso: tract-check eq <model.onnx> <in_feat.f32> <out.json> [batch]");
    }
    let onnx_path = &args[0];
    let input_path = &args[1];
    let out_json = &args[2];
    let batch: usize = if args.len() >= 4 {
        args[3].parse()?
    } else {
        1
    };
    let bytes = fs::read(onnx_path)?;
    let feat = read_f32_le(Path::new(input_path))?;
    if feat.len() != batch * 32 * 3 {
        bail!(
            "dimensões inválidas: esperado {} ({} x 32 x 3), obtido {}",
            batch * 32 * 3,
            batch,
            feat.len()
        );
    }
    let result = run_eq(&bytes, &feat, batch);
    let report = match &result {
        Ok(outputs) => json!({"batch": batch, "ok": true, "gains_db": outputs}),
        Err(e) => json!({"batch": batch, "ok": false, "error": format!("{e:#}")}),
    };
    fs::write(out_json, serde_json::to_string(&report)? + "\n")?;
    if let Err(e) = &result {
        eprintln!("tract recusou o modelo: {e:#}");
    }
    Ok(result.is_ok())
}
