#!/usr/bin/env python3
"""Compare annotated and disabled kernels using the existing Non-Spec simulator."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--gem5", type=Path, required=True)
    parser.add_argument("--sim-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    binaries = args.build_dir.resolve() / "bin"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tests = root / "llvm/test/CodeGen/RISCV"
    disables = ["-riscv-ns-disable-ir-loop-fission",
                "-riscv-ns-disable-software-pipelining",
                "-disable-loop-fission-decomposed-branch"]

    subprocess.run([str(binaries / "clang"),
                    "--target=riscv64-unknown-linux-gnu", "-march=rv64im", "-O2",
                    "-ffreestanding", "-fno-builtin", "-fno-stack-protector",
                    *[arg for flag in disables for arg in ("-mllvm", flag)],
                    "-c", str(tests / "Inputs/ns-loop-metadata-execution.c"),
                    "-o", str(output / "driver.o")], check=True)
    for mode in ("base", "annotated"):
        objects = [str(output / "driver.o")]
        for kind in ("fission", "lifetime", "inplace", "swp"):
            obj = output / (mode + "-" + kind + ".o")
            subprocess.run([str(binaries / "llc"),
                            "-mtriple=riscv64-unknown-linux-gnu", "-mattr=+m",
                            "-O2", "-filetype=obj",
                            *(disables if mode == "base" else []),
                            str(tests / ("ns-loop-metadata-" + kind + ".ll")),
                            "-o", str(obj)], check=True)
            objects.append(str(obj))
        binary = output / mode
        subprocess.run([str(binaries / "ld.lld"), "-m", "elf64lriscv", "-static",
                        "-e", "_start", *objects, "-o", str(binary)], check=True)
        run_dir = output / (mode + "-run")
        with (output / (mode + ".log")).open("w") as log:
            subprocess.run([str(args.gem5.resolve()), "-d", str(run_dir),
                            str(args.sim_script.resolve()), "--binary",
                            str(binary), "--argv0", "ns-loop-metadata",
                            "--stdout-file", str(output / (mode + ".stdout"))],
                           stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        status = json.loads((run_dir / "exit.json").read_text())
        if not status["program_exited"] or status["code"] != 0:
            raise SystemExit(f"{mode}: simulation failed: {status}")

    base = (output / "base.stdout").read_bytes()
    annotated = (output / "annotated.stdout").read_bytes()
    if len(base) != 84 * 8 or base != annotated:
        raise SystemExit("annotated results differ from baseline or are incomplete")
    print(f"84 results match; SHA256 {hashlib.sha256(base).hexdigest()}")


if __name__ == "__main__":
    main()
