//! Asset stateful pdfnet3-pro-v3-full no tract 0.19.16: os três grafos com forma fixa (S=1), hop a hop, realimentando
//! os pares `<nome>_in`/`<nome>_out` do manifest e roteando as saídas do enc para os decoders pelo nome. Roda os três
//! grafos em todo quadro (modo de paridade numérica, sem gating).
use std::{collections::HashMap, fs, path::Path};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;
use tract_onnx::prelude::*;

use crate::common::{read_f32_le, write_f32_le};

type Plan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

struct G {
    plan: Plan,
    inputs: Vec<(String, Vec<usize>)>,
    outputs: Vec<String>,
    pairs: Vec<(String, String)>,
    state: HashMap<String, Tensor>,
}

fn shape_of(v: &Value) -> Result<Vec<usize>> {
    v.as_array()
        .ok_or_else(|| anyhow!("shape inválido"))?
        .iter()
        .map(|d| {
            d.as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| anyhow!("dim inválida"))
        })
        .collect()
}

fn load(dir: &Path, spec: &Value) -> Result<G> {
    let file = spec["file"].as_str().ok_or_else(|| anyhow!("file"))?;
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
    for (k, v) in spec["inputs"]
        .as_object()
        .ok_or_else(|| anyhow!("inputs"))?
    {
        shapes.insert(k.clone(), shape_of(&v["shape"])?);
    }
    let mut pairs = Vec::new();
    let mut state = HashMap::new();
    for p in spec["state_pairs"]
        .as_array()
        .ok_or_else(|| anyhow!("state_pairs"))?
    {
        let (i, o) = (
            p["in"].as_str().unwrap().to_owned(),
            p["out"].as_str().unwrap().to_owned(),
        );
        let sh = shape_of(&p["shape"])?;
        state.insert(i.clone(), Tensor::zero::<f32>(&sh)?);
        shapes.insert(i.clone(), sh);
        pairs.push((i, o));
    }
    let mut model = onnx()
        .model_for_path(dir.join(file))
        .with_context(|| format!("lendo {file}"))?;
    let mut inputs = Vec::new();
    for (ix, outlet) in model.input_outlets()?.to_vec().iter().enumerate() {
        let name = model.node(outlet.node).name.clone();
        let sh = shapes
            .get(&name)
            .ok_or_else(|| anyhow!("entrada {name} fora do manifest"))?
            .clone();
        model.set_input_fact(ix, InferenceFact::dt_shape(f32::datum_type(), sh.clone()))?;
        inputs.push((name, sh));
    }
    let mut outputs = Vec::new();
    for outlet in model.output_outlets()?.to_vec() {
        outputs.push(
            model
                .outlet_label(outlet)
                .map(str::to_owned)
                .unwrap_or_else(|| model.node(outlet.node).name.clone()),
        );
    }
    let plan = model
        .into_optimized()
        .with_context(|| format!("otimizando {file}"))?
        .into_runnable()?;
    Ok(G {
        plan,
        inputs,
        outputs,
        pairs,
        state,
    })
}

impl G {
    fn step(&mut self, feeds: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let mut tv = TVec::new();
        for (name, sh) in &self.inputs {
            let t = self
                .state
                .get(name)
                .or_else(|| feeds.get(name))
                .ok_or_else(|| anyhow!("faltou {name}"))?;
            if t.shape() != sh.as_slice() {
                bail!("{name}: forma {:?} != {:?}", t.shape(), sh);
            }
            tv.push(t.clone().into());
        }
        let out = self.plan.run(tv)?;
        let mut map: HashMap<String, Tensor> = self
            .outputs
            .iter()
            .cloned()
            .zip(out.into_iter().map(|v| v.into_tensor()))
            .collect();
        for (i, o) in &self.pairs {
            let t = map
                .remove(o)
                .ok_or_else(|| anyhow!("saída de estado {o} ausente"))?;
            self.state.insert(i.clone(), t);
        }
        Ok(map)
    }
}

pub fn run(args: &[String]) -> Result<bool> {
    let [asset_dir, prefix] = args else {
        bail!("uso: tract-check stream <asset_dir> <prefix>  (lê prefix.{{feat_erb,feat_spec,speaker_embedding,speaker_present}}.f32)");
    };
    let dir = Path::new(asset_dir);
    let man: Value = serde_json::from_str(&fs::read_to_string(dir.join("manifest.json"))?)?;
    let mut enc = load(dir, &man["graphs"]["enc"])?;
    let mut erb = load(dir, &man["graphs"]["erb_dec"])?;
    let mut df = load(dir, &man["graphs"]["df_dec"])?;
    let rd = |k: &str| read_f32_le(Path::new(&format!("{prefix}.{k}.f32")));
    let (ferb, fspec, emb, present) = (
        rd("feat_erb")?,
        rd("feat_spec")?,
        rd("speaker_embedding")?,
        rd("speaker_present")?,
    );
    let frames = ferb.len() / 32;
    let cond: HashMap<String, Tensor> = HashMap::from([
        (
            "speaker_embedding".to_owned(),
            Tensor::from_shape(&[1, 192], &emb)?,
        ),
        (
            "speaker_present".to_owned(),
            Tensor::from_shape(&[1], &present)?,
        ),
    ]);
    let (mut m, mut coefs, mut lsnr) = (Vec::new(), Vec::new(), Vec::new());
    for t in 0..frames {
        let mut f = cond.clone();
        f.insert(
            "feat_erb".into(),
            Tensor::from_shape(&[1, 1, 1, 32], &ferb[t * 32..(t + 1) * 32])?,
        );
        f.insert(
            "feat_spec".into(),
            Tensor::from_shape(&[1, 2, 1, 96], &fspec[t * 192..(t + 1) * 192])?,
        );
        let e = enc.step(&f)?;
        let mut fe = cond.clone();
        for (k, v) in &e {
            fe.insert(k.clone(), v.clone());
        }
        m.extend_from_slice(erb.step(&fe)?["m"].as_slice::<f32>()?);
        coefs.extend_from_slice(df.step(&fe)?["coefs"].as_slice::<f32>()?);
        lsnr.extend_from_slice(e["lsnr"].as_slice::<f32>()?);
    }
    write_f32_le(Path::new(&format!("{prefix}.m.out.f32")), &m)?;
    write_f32_le(Path::new(&format!("{prefix}.coefs.out.f32")), &coefs)?;
    write_f32_le(Path::new(&format!("{prefix}.lsnr.out.f32")), &lsnr)?;
    Ok(true)
}
