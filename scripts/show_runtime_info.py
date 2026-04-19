#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata as metadata
import platform
import subprocess
import sys


def maybe_run(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        return completed.stdout.strip()
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"


def get_jax_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        normalized = name.lower()
        if normalized in {"jax", "jaxlib"} or normalized.startswith("jax-cuda"):
            packages[name] = dist.version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def main() -> None:
    packages = get_jax_packages()

    print(f"python: {sys.version.split()[0]}", flush=True)
    print(f"platform: {platform.platform()}", flush=True)
    if packages:
        print("jax packages:", flush=True)
        for name, version in packages.items():
            print(f"  - {name}=={version}", flush=True)

        cuda_plugins = [name for name in packages if name.endswith("-plugin")]
        if len(cuda_plugins) > 1:
            print("warning: multiple JAX CUDA plugin packages are installed in this environment.", flush=True)
            print("warning: keep exactly one CUDA extra, such as jax[cuda13] or jax[cuda12].", flush=True)

    import jax
    import jaxlib

    print(f"jax version: {jax.__version__}", flush=True)
    print(f"jaxlib version: {jaxlib.__version__}", flush=True)
    print(f"jax devices: {jax.devices()}", flush=True)
    print(
        f"nvidia-smi:\n{maybe_run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
