#!/usr/bin/env python3
"""Generate markdown tables summarizing IQA metrics.

This script now generates TWO markdown files in this directory:
- summary_tables_lambda.md: compares different lambda values for each metric
- summary_tables_methods.md: compares different planning methods for each metric

Usage: python generate_markdown_tables.py
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

# Methods for comparison tables (the last one maps to lambda_0_7)
METHODS = [
    "planned_no_optimisation",
    "planned_opt_no_verify",
    "random_planning",
    "planned_opt_verify",  # sourced from lambda_0_7 files
    "finetuned_planned_opt_verify",
]

# FR metrics (full-reference)
FR_METRICS = ["lpips", "psnr", "ssim"]
# NR metrics (no-reference) – we’ll look at the improvement (proc − raw)
NR_METRICS = ["qalign", "maniqa", "clipiqa", "musiq"]

# For convenience concatenate for unified processing where needed (we will store duplicates for proc & diff separately)
ALL_METRICS = FR_METRICS + NR_METRICS + [f"{m}_proc" for m in NR_METRICS]

# Metrics to include in the overall lambda summary table (averaged over degradations)
# We use the NR improvement (proc − raw) versions for NR metrics.
OVERVIEW_METRICS = [
    "psnr",
    "ssim",
    "lpips",
    "maniqa",
    "clipiqa",
    "musiq",
]

# Helper to map lambda float to filename-friendly string (replace '.' with '_')
def lambda_to_suffix(l: float) -> str:
    s = str(l)
    return s.replace(".", "_")


def load_lambda_metric_values() -> Dict[str, Dict[str, Dict[float, float]]]:
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


def load_method_metric_values() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load mean values for each degradation, metric, and method.

    Methods include:
    - planned_no_optimisation
    - planned_opt_no_verify
    - random_planning
    - planned_opt_verify (aliased from lambda_0_7 results)

    Returns
    -------
    dict
        {metric: {degradation: {method: mean_value}}}
    """
    data: Dict[str, Dict[str, Dict[str, float]]] = {m: {} for m in ALL_METRICS}
    for degradation in DEGRADATIONS:
        for method in METHODS:
            if method == "planned_opt_verify":
                # Use lambda_0_7 file as the source
                suffix = lambda_to_suffix(0.7)
                fname = f"{degradation}__lambda_{suffix}.json"
            else:
                fname = f"{degradation}__{method}.json"
            fpath = ROOT / fname
            if not fpath.is_file():
                raise FileNotFoundError(f"Missing JSON file: {fpath}")
            with fpath.open() as fp:
                content = json.load(fp)

            # Full-reference metrics: store mean directly
            for metric in FR_METRICS:
                key = f"FR_IQA.{metric}"
                mean_val = content[key]["mean"]
                data[metric].setdefault(degradation, {})[method] = mean_val

            # No-reference metrics: store the improvement (proc − raw) and processed-only value
            for metric in NR_METRICS:
                key_raw = f"NR_IQA.{metric}.raw"
                key_proc = f"NR_IQA.{metric}.proc"
                diff_val = content[key_proc]["mean"] - content[key_raw]["mean"]
                data[metric].setdefault(degradation, {})[method] = diff_val
                data[f"{metric}_proc"].setdefault(degradation, {})[method] = content[key_proc]["mean"]
    return data


# -----------------------------------------------------------------------------
# Markdown helpers
# -----------------------------------------------------------------------------


def format_lambda_table(metric: str, values: Dict[str, Dict[float, float]]) -> str:
    """Format markdown table for a single metric comparing lambdas."""
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


def format_lambda_overall_table(metrics_data: Dict[str, Dict[str, Dict[float, float]]]) -> str:
    """Create a markdown table averaging each metric across degradations for every lambda."""

    header = ["Lambda"] + [m.upper() for m in OVERVIEW_METRICS]
    lines: List[str] = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]

    # Rows per lambda
    for l in LAMBDAS:
        row_vals = []
        for metric in OVERVIEW_METRICS:
            # Mean over degradations for this metric & lambda
            mean_val = sum(metrics_data[metric][d][l] for d in DEGRADATIONS) / len(DEGRADATIONS)
            row_vals.append(f"{mean_val:.4f}")
        lines.append("| " + " | ".join([str(l)] + row_vals) + " |")

    return "\n".join(lines)


def format_method_overall_table(metrics_data: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """Create a markdown table averaging each metric across degradations for every method."""

    header = ["Method"] + [m.upper() for m in OVERVIEW_METRICS]
    lines: List[str] = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]

    for method in METHODS:
        row_vals = []
        for metric in OVERVIEW_METRICS:
            mean_val = sum(metrics_data[metric][d][method] for d in DEGRADATIONS) / len(DEGRADATIONS)
            row_vals.append(f"{mean_val:.4f}")
        lines.append("| " + " | ".join([method] + row_vals) + " |")

    return "\n".join(lines)


def format_method_table(metric: str, values: Dict[str, Dict[str, float]]) -> str:
    """Format markdown table for a single metric comparing methods."""
    header = ["Degradation \\ Method"] + METHODS
    lines: List[str] = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]

    # Rows per degradation
    for degradation in DEGRADATIONS:
        row_vals = [f"{values[degradation][m]:.4f}" for m in METHODS]
        lines.append("| " + " | ".join([degradation] + row_vals) + " |")

    # Column means across degradations
    col_means = []
    for m in METHODS:
        mean_m = sum(values[d][m] for d in DEGRADATIONS) / len(DEGRADATIONS)
        col_means.append(f"{mean_m:.4f}")
    lines.append("| " + " | ".join(["Mean"] + col_means) + " |")

    # Determine best method by column mean
    if metric == "lpips":
        best_idx = col_means.index(min(col_means, key=float))
        criterion = "minimum"
    else:
        best_idx = col_means.index(max(col_means, key=float))
        criterion = "maximum"
    best_method = METHODS[best_idx]

    lines.append("")
    lines.append(f"**Best method for {metric} (based on column {criterion}): {best_method}**")
    lines.append("")

    return "\n".join(lines)


def generate_lambda_markdown() -> str:
    metrics_data = load_lambda_metric_values()
    md_lines: List[str] = []

    # Overall summary table first
    md_lines.append("## Average scores across degradations\n")
    md_lines.append(format_lambda_overall_table(metrics_data))
    md_lines.append("")

    # FR metrics first
    for metric in FR_METRICS:
        md_lines.append(f"### {metric.upper()}\n")
        md_lines.append(format_lambda_table(metric, metrics_data[metric]))

    # NR metrics (improvement)
    for metric in NR_METRICS:
        md_lines.append(f"### Δ {metric.upper()} (proc − raw)\n")
        md_lines.append(format_lambda_table(metric, metrics_data[metric]))

    # NR metrics (processed absolute)
    for metric in NR_METRICS:
        tag = f"{metric}_proc"
        md_lines.append(f"### {metric.upper()} (processed)\n")
        md_lines.append(format_lambda_table(tag, metrics_data[tag]))

    return "\n".join(md_lines)


def generate_methods_markdown() -> str:
    metrics_data = load_method_metric_values()
    md_lines: List[str] = []

    # Overall summary table first
    md_lines.append("## Average scores across degradations\n")
    md_lines.append(format_method_overall_table(metrics_data))
    md_lines.append("")

    # FR metrics first
    for metric in FR_METRICS:
        md_lines.append(f"### {metric.upper()}\n")
        md_lines.append(format_method_table(metric, metrics_data[metric]))

    # NR metrics (improvement)
    for metric in NR_METRICS:
        md_lines.append(f"### Δ {metric.upper()} (proc − raw)\n")
        md_lines.append(format_method_table(metric, metrics_data[metric]))

    # NR metrics (processed absolute)
    for metric in NR_METRICS:
        tag = f"{metric}_proc"
        md_lines.append(f"### {metric.upper()} (processed)\n")
        md_lines.append(format_method_table(tag, metrics_data[tag]))

    return "\n".join(md_lines)


def main() -> None:
    # Generate lambda comparison tables
    markdown_lambda = generate_lambda_markdown()
    out_lambda = ROOT / "summary_tables_lambda.md"
    out_lambda.write_text(markdown_lambda)

    # Generate method comparison tables
    markdown_methods = generate_methods_markdown()
    out_methods = ROOT / "summary_tables_methods.md"
    out_methods.write_text(markdown_methods)

    print(f"Markdown tables saved to {out_lambda} and {out_methods}")


if __name__ == "__main__":
    main()
