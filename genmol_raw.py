#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genmol_raw.py - minimal S4 molecule generator with chunked sampling.

Generates raw SMILES strings with optional prefix at a given sampling temperature.
No filtering, metrics, or predictors are applied; the raw outputs (plus mean log-likelihood)
are written directly to CSV.
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from s4dd.s4_for_denovo_design import S4forDenovoDesign


MAX_GEN_PER_CALL = 1000  # safety cap to avoid oversized single-call sampling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate raw SMILES from an S4 model without any filtering or prediction."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing the trained S4 model.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=5000,
        help="Number of molecules to generate (default: 5000).",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=2.0,
        help="Sampling temperature for the generator (default: 2.0).",
    )
    parser.add_argument(
        "--required-prefix-smiles",
        type=str,
        default=None,
        help="Optional prefix SMILES passed to S4 (set to empty for no prefix).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="molecules_raw.csv",
        help="Output CSV path for raw SMILES (and mean log-likelihood).",
    )
    return parser.parse_args()


def load_generator(model_dir: str) -> S4forDenovoDesign:
    print(f"Loading S4 model from {model_dir} ...")
    generator = S4forDenovoDesign.from_file(model_dir)
    print("Model loaded.")
    return generator


def generate_smiles(
    generator: S4forDenovoDesign, num: int, temp: float, prefix: str
) -> Tuple[List[str], List[float]]:
    if num <= 0:
        return [], []

    all_smiles: List[str] = []
    all_log_likelihoods: List[float] = []
    prefix_display = prefix if prefix else "None"
    print(
        f"Sampling {num} molecules in chunks (max {MAX_GEN_PER_CALL} per call) "
        f"(temp={temp}, prefix={prefix_display}) ..."
    )

    with tqdm(total=num, desc="生成进度") as pbar:
        while len(all_smiles) < num:
            remaining = num - len(all_smiles)
            cur_n = min(remaining, MAX_GEN_PER_CALL)

            smiles_chunk, ll_chunk = generator.design_molecules(
                n_designs=cur_n,
                batch_size=cur_n,
                temperature=temp,
                required_prefix_smiles=prefix or None,
            )

            if not smiles_chunk:
                raise RuntimeError(
                    "design_molecules returned no molecules; aborting to avoid an infinite loop."
                )

            if len(smiles_chunk) != len(ll_chunk):
                print(
                    "Warning: chunk length mismatch between SMILES and likelihoods; truncating this chunk."
                )

            chunk_len = min(len(smiles_chunk), len(ll_chunk))
            all_smiles.extend(smiles_chunk[:chunk_len])
            all_log_likelihoods.extend(ll_chunk[:chunk_len])
            pbar.update(chunk_len)

    if len(all_smiles) > num:
        all_smiles = all_smiles[:num]
        all_log_likelihoods = all_log_likelihoods[:num]

    return all_smiles, all_log_likelihoods


def write_csv(path: Path, smiles: List[str], log_likelihoods: List[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    header = ["smiles", "mean_log_likelihood"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for smi, ll in zip(smiles, log_likelihoods):
            writer.writerow([smi, ll])


def main() -> None:
    args = parse_args()

    try:
        generator = load_generator(args.model_dir)
    except Exception as exc:
        print(f"Failed to load S4 model: {exc}")
        return

    try:
        smiles, log_likelihoods = generate_smiles(
            generator=generator,
            num=args.num,
            temp=args.temp,
            prefix=args.required_prefix_smiles,
        )
    except Exception as exc:
        print(f"Generation failed: {exc}")
        return

    if len(log_likelihoods) != len(smiles):
        print(
            "Warning: output length mismatch between SMILES and likelihoods; truncating to the shorter list."
        )
        limit = min(len(smiles), len(log_likelihoods))
        smiles = smiles[:limit]
        log_likelihoods = log_likelihoods[:limit]

    output_path = Path(args.output_csv)
    write_csv(output_path, smiles, log_likelihoods)
    print(f"Wrote {len(smiles)} molecules to {output_path.resolve()}")


if __name__ == "__main__":
    main()
