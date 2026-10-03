"""Amostragem do nvidia-smi a cada 1 s durante uma medição (temperatura, clock, potência, motivos de throttling)."""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass

FIELDS = ["timestamp", "temperature.gpu", "clocks.sm", "power.draw", "memory.used"]
REASON_FIELDS = ["clocks_event_reasons.active", "clocks_throttle_reasons.active"]


@dataclass(frozen=True)
class GpuSample:
    temperature_c: float
    sm_clock_mhz: float
    power_w: float
    memory_used_mib: float
    reasons: str


def _num(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return float("nan")


def parse_line(line: str) -> GpuSample:
    parts = [p.strip() for p in line.split(",")]
    return GpuSample(_num(parts[1]), _num(parts[2]), _num(parts[3]), _num(parts[4]), parts[5])


def _query() -> list[str]:
    for reason in REASON_FIELDS:
        fields = FIELDS + [reason]
        probe = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, check=False)
        if probe.returncode == 0:
            return fields
    raise RuntimeError("nvidia-smi não aceitou nenhum campo de motivo de throttling")


class GpuMonitor:
    def __init__(self, interval_ms: int = 1000):
        self.interval_ms = interval_ms
        self.samples: list[GpuSample] = []

    def __enter__(self) -> GpuMonitor:
        fields = _query()
        self._proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits",
             f"-lms={self.interval_ms}"], stdout=subprocess.PIPE, text=True)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self) -> None:
        for line in self._proc.stdout:
            if line.strip():
                self.samples.append(parse_line(line))

    def __exit__(self, *exc) -> None:
        self._proc.terminate()
        self._proc.wait(timeout=10)
        self._thread.join(timeout=10)

    def summary(self) -> dict:
        s = self.samples
        reasons = sorted({x.reasons for x in s if x.reasons and int(x.reasons, 16) != 0}) if s else []
        return {
            "samples": len(s),
            "max_temperature_c": max((x.temperature_c for x in s), default=float("nan")),
            "min_sm_clock_mhz": min((x.sm_clock_mhz for x in s), default=float("nan")),
            "mean_power_w": float(sum(x.power_w for x in s) / len(s)) if s else float("nan"),
            "max_memory_used_mib": max((x.memory_used_mib for x in s), default=float("nan")),
            "nonzero_throttle_reasons": reasons,
        }
