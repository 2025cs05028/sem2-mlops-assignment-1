"""
Run the heart-disease pipeline end-to-end with a single command.

Usage:
  python run_pipeline.py
  python run_pipeline.py --skip-training
  python run_pipeline.py --with-tests
  python run_pipeline.py --verbose
  python run_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(command: list[str], dry_run: bool, verbose: bool) -> None:
    print("$ " + " ".join(command))
    if dry_run:
        return
    started = time.time()
    if verbose:
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        try:
            out = subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            if exc.stdout:
                print(exc.stdout)
            if exc.stderr:
                print(exc.stderr)
            raise
        if out.stdout.strip():
            last_line = out.stdout.strip().splitlines()[-1]
            print(f"  -> {last_line}")
    print(f"  -> done in {time.time() - started:.1f}s\n")


def build_steps(skip_training: bool, with_tests: bool) -> list[list[str]]:
    steps: list[list[str]] = [
        [sys.executable, "fetch_raw_data.py"],
        [sys.executable, "dataCleaning.py"],
        [sys.executable, "featureEngineering.py"],
    ]
    if not skip_training:
        steps.append([sys.executable, "modelTraining.py"])
    steps.extend(
        [
            [sys.executable, "modelSelection.py"],
            [sys.executable, "package_production_model.py"],
        ]
    )
    if with_tests:
        steps.append([sys.executable, "-m", "pytest", "tests/", "-q"])
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full mlOps pipeline from one command.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip modelTraining.py baseline stage.",
    )
    parser.add_argument(
        "--with-tests",
        action="store_true",
        help="Run pytest suite after packaging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full output from each stage (default is compact).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Project root: {ROOT}")
    for step in build_steps(args.skip_training, args.with_tests):
        _run(step, dry_run=args.dry_run, verbose=args.verbose)
    if args.dry_run:
        print("Dry run complete.")
    else:
        print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
