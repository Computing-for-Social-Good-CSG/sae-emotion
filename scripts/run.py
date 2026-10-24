#!/usr/bin/env python3
"""Run a preserved research entry point with consistent imports and cwd."""
from pathlib import Path
import argparse
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "prepare-lexical-control": "experiments/evaluation/prepare_lexical_control.py",
    "extract-activations": "experiments/rq1_features/extract_activations.py",
    "select-candidates": "experiments/rq1_features/select_candidates.py",
    "estimate-effects": "experiments/rq1_features/estimate_causal_effects.py",
    "ablate-features": "experiments/rq2_causal/ablate_features.py",
    "patch-activations": "experiments/rq2_causal/patch_activations.py",
    "adjacent-mediation": "experiments/rq2_causal/adjacent_layer_mediation.py",
    "targeted-amplification": "experiments/rq3_steering/targeted_amplification.py",
    "train-steering": "experiments/rq3_steering/train_global_steering.py",
    "general-capabilities": "experiments/evaluation/general_capabilities.py",
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=STAGES)
    parser.add_argument("--list", action="store_true", help="List entry points without importing research dependencies")
    args = parser.parse_args()
    if args.list or not args.stage:
        for name, path in STAGES.items():
            print(f"{name:24} {path}")
        return
    script = ROOT / STAGES[args.stage]
    print(f"Running {script.relative_to(ROOT)} with its saved configuration.", flush=True)
    print("Review the model, SAE, and run settings at the top of that script.", flush=True)
    sys.path.insert(0, str(ROOT / "src"))
    os.chdir(ROOT)
    sys.argv = [str(script)]
    runpy.run_path(str(script), run_name="__main__")

if __name__ == "__main__":
    main()
