#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from alphagateau_breakthrough.plotting import generate_submission_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the three paper figures from experiment outputs.")
    parser.add_argument("--gnn-scratch-dir", required=True)
    parser.add_argument("--pretrain-dir", required=True)
    parser.add_argument("--finetune-dir", required=True)
    parser.add_argument("--output-dir", default="report/figures")
    args = parser.parse_args()
    generate_submission_figures(
        gnn_scratch_dir=args.gnn_scratch_dir,
        pretrain_dir=args.pretrain_dir,
        finetune_dir=args.finetune_dir,
        output_dir=args.output_dir,
    )
    print({"output_dir": args.output_dir})


if __name__ == "__main__":
    main()
