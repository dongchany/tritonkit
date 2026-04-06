"""Hardware fingerprinting for benchmark reproducibility."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import torch


@dataclass
class HardwareFingerprint:
    gpu_name: str
    compute_capability: str
    driver_version: str
    memory_gb: float
    sm_count: int
    sm_clock_mhz: int
    mem_clock_mhz: int
    pcie_gen: str
    peak_bandwidth_gbps: float
    peak_tflops_fp16: float

    def to_dict(self) -> dict:
        return {
            "gpu_name": self.gpu_name,
            "compute_capability": self.compute_capability,
            "driver_version": self.driver_version,
            "memory_gb": self.memory_gb,
            "sm_count": self.sm_count,
            "sm_clock_mhz": self.sm_clock_mhz,
            "mem_clock_mhz": self.mem_clock_mhz,
            "pcie_gen": self.pcie_gen,
            "peak_bandwidth_gbps": round(self.peak_bandwidth_gbps, 2),
            "peak_tflops_fp16": round(self.peak_tflops_fp16, 2),
        }


def detect_hardware() -> HardwareFingerprint:
    """Auto-detect current GPU hardware."""
    props = torch.cuda.get_device_properties(0)

    driver = "unknown"
    sm_clock = 0
    mem_clock = 0
    pcie_gen = "unknown"

    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,clocks.sm,clocks.mem,pcie.link.gen.current",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if smi.returncode == 0:
            parts = smi.stdout.strip().split(", ")
            if len(parts) >= 4:
                driver = parts[0].strip()
                sm_clock = int(parts[1].strip())
                mem_clock = int(parts[2].strip())
                pcie_gen = parts[3].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Defer peak measurements to avoid circular import
    from tritonkit.bench.roofline import measure_peak_bandwidth, measure_peak_flops

    return HardwareFingerprint(
        gpu_name=props.name,
        compute_capability=f"{props.major}.{props.minor}",
        driver_version=driver,
        memory_gb=round(props.total_mem / 1e9, 2),
        sm_count=props.multi_processor_count,
        sm_clock_mhz=sm_clock,
        mem_clock_mhz=mem_clock,
        pcie_gen=pcie_gen,
        peak_bandwidth_gbps=measure_peak_bandwidth(),
        peak_tflops_fp16=measure_peak_flops(torch.float16),
    )
