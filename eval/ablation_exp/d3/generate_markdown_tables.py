#!/usr/bin/env python3
"""Generate markdown tables summarizing FR IQA metrics across degradations and lambda values.

Usage: python generate_markdown_tables.py > summary_tables.md
"""
import json
from pathlib import Path
from typing import Dict, List

# Directory containing the JSON results (assumed to be this script's directory)
ROOT = Path(__file__).resolve().parent

# Configuration
DEGRADATIONS = [
    "dark_defocus_blur_jpeg compression_artifact",
    "haze_motion_blur_low_resolution",
    "motion_blur_defocus_blur_noise",
    "rain_noise_low_resolution",
]
LAMBDAS = [0, 0.3, 0.5, 0.7]

# FR metrics (full-reference)
FR_METRICS = ["lpips", "psnr", "ssim"]
# NR metrics (no-reference) – we’ll look at the improvement (proc − raw)
NR_METRICS = ["qalign", "maniqa", "clipiqa", "musiq"]

# For convenience concatenate for unified processing where needed (we will store duplicates for proc & diff separately)
ALL_METRICS = FR_METRICS + NR_METRICS + [f"{m}_proc" for m in NR_METRICS]

# Helper to map lambda float to filename-friendly string (replace '.' with '_')
def lambda_to_suffix(l: float) -> str:
    s = str(l)
    return s.replace(".", "_")


def load_metric_values() -> Dict[str, Dict[str, Dict[float, float]]]:
    """Load mean values for each degradation, metric, and lambda.

    Returns
    -------
    dict
        {metric: {degradation: {lambda: mean_value}}}
    """
    data: Dict[str, Dict[str, Dict[float, float]]] = {m: {} for m in ALL_METRICS}
    for degradation in DEGRADATIONS:
        for l in LAMBDAS:
            suffix = lambda_to_suffix(l)
            fname = f"{degradation}__lambda_{suffix}.json"
            fpath = ROOT / fname
            if not fpath.is_file():
                raise FileNotFoundError(f"Missing JSON file: {fpath}")
            with fpath.open() as fp:
                content = json.load(fp)
            # Full-reference metrics: store mean directly
            for metric in FR_METRICS:
                key = f"FR_IQA.{metric}"
                mean_val = content[key]["mean"]
                data[metric].setdefault(degradation, {})[l] = mean_val

            # No-reference metrics: store the improvement (proc − raw) and processed-only value
            for metric in NR_METRICS:
                key_raw = f"NR_IQA.{metric}.raw"
                key_proc = f"NR_IQA.{metric}.proc"
                diff_val = content[key_proc]["mean"] - content[key_raw]["mean"]
                data[metric].setdefault(degradation, {})[l] = diff_val
                data[f"{metric}_proc"].setdefault(degradation, {})[l] = content[key_proc]["mean"]
    return data


# -----------------------------------------------------------------------------
# Markdown helpers
# -----------------------------------------------------------------------------


def format_table(metric: str, values: Dict[str, Dict[float, float]]) -> str:
    """Format markdown table for a single metric."""
    header = ["Degradation \\ Lambda"] + [str(l) for l in LAMBDAS]
    lines: List[str] = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]

    # Rows per degradation
    for degradation in DEGRADATIONS:
        row_vals = [f"{values[degradation][l]:.4f}" for l in LAMBDAS]
        lines.append("| " + " | ".join([degradation] + row_vals) + " |")

    # Column means
    col_means = []
    for l in LAMBDAS:
        mean_l = sum(values[d][l] for d in DEGRADATIONS) / len(DEGRADATIONS)
        col_means.append(f"{mean_l:.4f}")
    lines.append("| " + " | ".join(["Mean"] + col_means) + " |")

    # Determine ideal lambda
    if metric == "lpips":
        # lower is better
        ideal_lambda = LAMBDAS[col_means.index(min(col_means, key=float))]
        criterion = "minimum"
    else:
        # higher is better for all other metrics (PSNR, SSIM, and NR diffs)
        ideal_lambda = LAMBDAS[col_means.index(max(col_means, key=float))]
        criterion = "maximum"

    lines.append("")
    lines.append(f"**Ideal lambda for {metric} (based on column {criterion}): {ideal_lambda}**")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    metrics_data = load_metric_values()

    md_lines: List[str] = []

    # FR metrics first
    for metric in FR_METRICS:
        md_lines.append(f"### {metric.upper()}\n")
        md_lines.append(format_table(metric, metrics_data[metric]))

    # NR metrics (improvement)
    for metric in NR_METRICS:
        md_lines.append(f"### Δ {metric.upper()} (proc − raw)\n")
        md_lines.append(format_table(metric, metrics_data[metric]))

    # NR metrics (processed absolute)
    for metric in NR_METRICS:
        tag = f"{metric}_proc"
        md_lines.append(f"### {metric.upper()} (processed)\n")
        md_lines.append(format_table(tag, metrics_data[tag]))

    markdown_text = "\n".join(md_lines)

    # Print to stdout
    print(markdown_text)

    # Also save to file in same directory
    out_path = ROOT / "summary_tables.md"
    out_path.write_text(markdown_text)
    print(f"\nMarkdown tables saved to {out_path}")


if __name__ == "__main__":
    main()
