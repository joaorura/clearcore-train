mod common;
mod enroll;
mod gating;
mod identity;

use std::process::ExitCode;

const USAGE: &str = "uso: tract-check identity <film_asset.tar.gz> <approved_asset.bin> <golden.f32> <out.json>
       tract-check gating <asset.tar.gz> <in_48k.f32> <out_prefix> [film.f32]
       tract-check enroll <model.onnx> <in_16k.f32> <out.json>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("identity") => identity::run(&args[1..]),
        Some("gating") => gating::run(&args[1..]),
        Some("enroll") => enroll::run(&args[1..]),
        _ => Err(anyhow::anyhow!(USAGE)),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("erro: {e:#}");
            ExitCode::from(2)
        }
    }
}
