# Aggregates inference-speed numbers (attempt1) across d3 sub-folders and
# prints a markdown table with mean ± std for each method.
#
# Folder structure expected:
# eval/data/d3/{folder}/{method}/artefacts/inference_speed.json
#
# Each JSON file must have the shape
#   {
#     "attempt1": {
#         "planner.batching": float,
#         "planner.optimization": float,
#         "executor": float,
#         "verifier": float
#     }
#   }
#
# For the method "planned_opt_verify" the user provided fixed numbers that
# override whatever may be on disk.

from pathlib import Path
from statistics import mean, stdev, StatisticsError
import json

from config import settings  # project-wide BASE path

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

FOLDERS = [
    "dark_defocus_blur_jpeg compression_artifact",
    "haze_motion_blur_low_resolution",
    "motion_blur_defocus_blur_noise",
    "rain_noise_low_resolution",
]

METHODS = [
    "planned_no_optimisation",
    "planned_opt_no_verify",
    "planned_opt_verify",
    "finetuned_planned_opt_verify",
    "random_planning",
]

# Constant numbers for planned_opt_verify (avg across images, seconds)
USER_FIXED_PLANNED_OPT_VERIFY = {
    "planner.batching": 275.83,
    "planner.optimization": 5856.21,
    "executor": 5313.63,
    "verifier": 2752.6,
}

COMPONENTS = [
    "planner.batching",
    "planner.optimization",
    "executor",
    "verifier",
]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def safe_stdev(values):
    """Return stdev(values) or 0.0 when <2 samples."""

    try:
        return stdev(values)
    except StatisticsError:
        return 0.0


# ---------------------------------------------------------------------------
# Collect numbers
# ---------------------------------------------------------------------------

data = {m: {c: [] for c in COMPONENTS + ["total"]} for m in METHODS}

for folder in FOLDERS:
    for method in METHODS:
        if method == "planned_opt_verify":
            comp_values = USER_FIXED_PLANNED_OPT_VERIFY
        else:
            json_path = (
                Path(settings.BASE)
                / f"eval/data/d3/{folder}/{method}/artefacts/inference_speed.json"
            )

            if not json_path.exists():
                # Skip missing – leave empty list, warn in console.
                print(f"[WARN] Missing {json_path} – skipping.")
                continue

            with open(json_path, "r") as fh:
                try:
                    comp_values = json.load(fh)["attempt1"]
                except (json.JSONDecodeError, KeyError) as err:
                    print(f"[WARN] Malformed {json_path}: {err} – skipping.")
                    continue

        # -------------------------------------------------------------------
        # Convert to per-image numbers (each folder has 100 images)
        # -------------------------------------------------------------------

        comp_values = {k: v / 100 for k, v in comp_values.items()}

        total = (
            comp_values["planner.batching"]
            + comp_values["planner.optimization"]
            + comp_values["executor"]
            + comp_values["verifier"]
        )

        # accumulate
        for c in COMPONENTS:
            data[method][c].append(comp_values[c])
        data[method]["total"].append(total)

# ---------------------------------------------------------------------------
# Build markdown table
# ---------------------------------------------------------------------------


def fmt(values):
    """Format as mean ± std with 2 decimals."""

    if not values:
        return "—"  # em dash for missing data
    return f"{mean(values):.2f} ± {safe_stdev(values):.2f}"


header = (
    "| method | "
    + " | ".join(COMPONENTS)
    + " | total |\n| --- | "
    + " | ".join(["---"] * (len(COMPONENTS) + 1))
    + " |"
)

rows = []
for method in METHODS:
    row_cells = [method]
    for c in COMPONENTS + ["total"]:
        row_cells.append(fmt(data[method][c]))
    rows.append("| " + " | ".join(row_cells) + " |")

markdown_table = "\n".join([header] + rows)

print("\nGenerated markdown table:\n")
print(markdown_table)

# Optionally save to file next to this script
md_path = Path(__file__).with_suffix(".md")
md_path.write_text(markdown_table)
print(f"\nTable saved to {md_path}")
        

