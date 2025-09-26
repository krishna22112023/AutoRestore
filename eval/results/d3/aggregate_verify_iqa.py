#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, List, Tuple
import math

# Roots
THIS_DIR = Path(__file__).resolve().parent
BASE = THIS_DIR.parents[2]  # /home/.../AutoRestore
DATA_ROOT = BASE / "eval/data/d3"
OUT_ROOT = THIS_DIR  # write summaries next to this script

DEGRADATIONS: List[str] = [
    "rain_noise_low_resolution",
]
LAMBDA_SUFFIXES: List[str] = ["lambda_0", "lambda_0_3", "lambda_0_5", "lambda_0_7"]

FR_KEYS: List[str] = ["lpips", "psnr", "ssim"]
#NR_KEYS: List[str] = ["qalign", "maniqa", "clipiqa", "musiq"]


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / n
    return m, math.sqrt(var)


def aggregate_one(verify_path: Path) -> Dict[str, Dict[str, float]]:
    with verify_path.open() as fp:
        verify = json.load(fp)

    # Collectors
    fr_values: Dict[str, List[float]] = {k: [] for k in FR_KEYS}
    #nr_raw_values: Dict[str, List[float]] = {k: [] for k in NR_KEYS}
    #nr_proc_values: Dict[str, List[float]] = {k: [] for k in NR_KEYS}

    for _img, rec in verify.items():
        # FR
        if "FR_IQA" in rec:
            fr = rec["FR_IQA"]
            for k in FR_KEYS:
                if k in fr and fr[k] is not None:
                    fr_values[k].append(float(fr[k]))
        # NR
        '''if "NR_IQA" in rec:
            nr = rec["NR_IQA"]
            for k in NR_KEYS:
                if k in nr:
                    raw = nr[k].get("raw")
                    proc = nr[k].get("proc")
                    if raw is not None:
                        nr_raw_values[k].append(float(raw))
                    if proc is not None:
                        nr_proc_values[k].append(float(proc))'''

    # Build FR-only summary dict
    summary: Dict[str, Dict[str, float]] = {}

    #for k in NR_KEYS:
    #    m, s = mean_std(nr_raw_values[k])
    #    summary[f"NR_IQA.{k}.raw"] = {"mean": m, "std": s}
    #    m, s = mean_std(nr_proc_values[k])
    #    summary[f"NR_IQA.{k}.proc"] = {"mean": m, "std": s}

    for k in FR_KEYS:
        m, s = mean_std(fr_values[k])
        summary[f"FR_IQA.{k}"] = {"mean": m, "std": s}

    return summary


def merge_preserve_nr(existing: Dict[str, Dict[str, float]], fr_summary: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Merge FR_IQA values into existing dict, preserving any keys starting with 'NR_IQA.'"""
    merged = dict(existing) if isinstance(existing, dict) else {}
    for k, v in fr_summary.items():
        merged[k] = v
    return merged


def main() -> None:
    generated = 0
    missing = []
    for deg in DEGRADATIONS:
        for lam in LAMBDA_SUFFIXES:
            in_path = DATA_ROOT / deg / lam / "artefacts" / "verify_IQA.json"
            if not in_path.is_file():
                missing.append(str(in_path))
                continue
            fr_summary = aggregate_one(in_path)
            out_name = f"{deg}__{lam}.json"
            out_path = OUT_ROOT / out_name

            # Load existing (to preserve NR_IQA) if present
            existing: Dict[str, Dict[str, float]] = {}
            if out_path.is_file():
                try:
                    with out_path.open() as fp:
                        existing = json.load(fp)
                except Exception:
                    existing = {}

            merged = merge_preserve_nr(existing, fr_summary)

            with out_path.open("w") as fp:
                json.dump(merged, fp, indent=2)
            print(f"Wrote {out_path}")
            generated += 1

    if missing:
        print("Missing verify_IQA.json files:")
        for p in missing:
            print(f" - {p}")
    print(f"Generated {generated} summary files.")


if __name__ == "__main__":
    main()
