//! Carga do enrollment como `crates/model/src/enrollment.rs:418-435`: entrada 0 fixada em f32 [1, N].
use std::{collections::BTreeMap, fs, io::Cursor, path::Path};

use anyhow::{bail, Result};
use serde_json::json;
use tract_onnx::prelude::*;

use crate::common::read_f32_le;

fn extract(bytes: &[u8], audio: &[f32]) -> TractResult<BTreeMap<String, Vec<f32>>> {
    let mut model = onnx().model_for_read(&mut Cursor::new(bytes))?;
    let mut names = Vec::new();
    for outlet in model.output_outlets()?.to_vec() {
        names.push(model.outlet_label(outlet).map(str::to_owned));
    }
    model.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, audio.len())))?;
    let runnable = model.into_optimized()?.into_runnable()?;
    let input = Tensor::from_shape(&[1, audio.len()], audio)?;
    let outputs = runnable.run(tvec!(input.into()))?;
    let mut map = BTreeMap::new();
    for (name, value) in names.iter().zip(outputs.iter()) {
        if let Some(name) = name {
            map.insert(name.clone(), value.as_slice::<f32>()?.to_vec());
        }
    }
    Ok(map)
}

pub fn run(args: &[String]) -> Result<bool> {
    let [onnx_path, input_path, out_json] = args else {
        bail!("uso: tract-check enroll <model.onnx> <in_16k.f32> <out.json>");
    };
    let bytes = fs::read(onnx_path)?;
    let audio = read_f32_le(Path::new(input_path))?;
    let result = extract(&bytes, &audio);
    let report = match &result {
        Ok(outputs) => json!({"n": audio.len(), "ok": true, "outputs": outputs}),
        Err(e) => json!({"n": audio.len(), "ok": false, "error": format!("{e:#}")}),
    };
    fs::write(out_json, serde_json::to_string(&report)? + "\n")?;
    if let Err(e) = &result {
        eprintln!("tract recusou o modelo: {e:#}");
    }
    Ok(result.is_ok())
}
