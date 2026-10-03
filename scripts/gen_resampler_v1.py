"""Gera resampler/v1: coeficientes f32, SHA-256 e vetores dourados (determinístico)."""

import json

from cctrain.data import resample as rs
from cctrain.hashing import sha256_bytes


def main() -> None:
    out = rs.V1_DIR
    (out / "golden").mkdir(parents=True, exist_ok=True)
    h = rs.design_coefficients().astype("<f4")
    (out / "coeffs.f32").write_bytes(h.tobytes())
    coeffs_sha = sha256_bytes(h.tobytes())
    (out / "coeffs.sha256").write_text(f"{coeffs_sha}  coeffs.f32\n")
    golden = {}
    for name, x in rs.golden_inputs().items():
        y = rs.resample_48k_to_16k(x, h)
        (out / "golden" / f"{name}.in.f32").write_bytes(x.astype("<f4").tobytes())
        (out / "golden" / f"{name}.out.f32").write_bytes(y.astype("<f4").tobytes())
        golden[name] = {"n_in": int(len(x)), "n_out": int(len(y)),
                        "in_sha256": sha256_bytes(x.astype("<f4").tobytes()),
                        "out_sha256": sha256_bytes(y.astype("<f4").tobytes())}
    manifest = {
        "name": rs.NAME,
        "formula": "y[n] = sum_{k=0}^{240} h[k] * x[3n + 120 - k]; x = 0 fora de [0, L); len(y) = ceil(L/3)",
        "design": "h[k] = 2fc*sinc(2fc*(k-120)) * kaiser(241, 8.6)[k], fc = 7600/48000, normalizado para soma 1 em float64",
        "taps": rs.TAPS, "decimation": rs.DECIMATION, "fs_in": rs.FS_IN, "fs_out": rs.FS_OUT,
        "cutoff_hz": rs.CUTOFF_HZ, "kaiser_beta": rs.KAISER_BETA,
        "coefficients": {"file": "coeffs.f32", "dtype": "float32-le", "sha256": coeffs_sha},
        "arithmetic": "acumulação em float64 sobre os coeficientes f32; saída arredondada para float32",
        "golden_tolerance_abs_proposed": 1.0e-6,
        "golden": golden,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(coeffs_sha)


if __name__ == "__main__":
    main()
