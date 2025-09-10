import json
import shutil
import subprocess
import sys
from statistics import mean, stdev
from pathlib import Path
from typing import Dict, List, Any
import yaml

from config import logger, settings

ROOT = settings.BASE
BASE_YAML = ROOT / "config/base.yaml"
RUN_SCRIPT = ROOT / "run.py"

D3_ROOT = ROOT / "eval/data/d3"
RESULTS_ROOT = ROOT / "eval/ablation_exp/d3"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

def load_experiment_names() -> List[str]:
    """Return experiment names defined in base.yaml (excluding the *base* template)."""
    with BASE_YAML.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    logger.info("Loaded base.yaml to extract experiment names. Found %d experiments (excluding base).", len(raw.get("experiments", {})) - 1)
    experiments = raw.get("experiments", {})
    return [name for name in experiments.keys() if name != "base"]


def create_temp_config(base_cfg: Dict[str, Any], experiment_name: str, sub_raw: Path,
                        sub_processed: Path, sub_artefacts: Path) -> Path:
    """Create a temporary YAML config overriding data paths for a specific sub-folder."""
    temp_cfg = {
        "experiments": {
            # retain the base template (deep-copy) and overrides for experiment
            "base": base_cfg["base"],
            experiment_name: base_cfg[experiment_name],
        }
    }
    # Override paths inside base template so that anchoring works for all variants
    # We modify both *base* and the target experiment, as other experiments inherit.
    for key in ("base", experiment_name):
        general = temp_cfg["experiments"][key].setdefault("general", {})
        general.update({
            "data_path": str(sub_raw.relative_to(ROOT)),
            "processed_path": str(sub_processed.relative_to(ROOT)),
            "artefacts_path": str(sub_artefacts.relative_to(ROOT)),
        })

    temp_path = sub_artefacts.parent / f"temp_{experiment_name}.yaml"
    with temp_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(temp_cfg, fh)
    return temp_path


def move_images_into_raw(subfolder: Path, exts: List[str]) -> Path:
    """Move png images in *subfolder* into a *raw* directory.

    Returns the path to the *raw* directory."""
    raw_dir = subfolder / "raw"
    raw_dir.mkdir(exist_ok=True)
    patterns = [f"*.{e}" for e in exts]
    for pattern in patterns:
        for img in subfolder.glob(pattern):
            shutil.move(str(img), raw_dir / img.name)
            logger.info("Moved %s to %s", img.name, raw_dir)
    return raw_dir


def run_experiment(config_path: Path, experiment_name: str):
    """Invoke *run.py* for the given config & experiment."""
    logger.info("Running experiment '%s' for config %s", experiment_name, config_path)
    cmd = [sys.executable, str(RUN_SCRIPT), "--config_path", str(config_path), "--experiment", experiment_name]
    subprocess.run(cmd, check=True)
    logger.info("Completed experiment '%s' for config %s", experiment_name, config_path)


def aggregate_verify_scores(verify_path: Path) -> Dict[str, Dict[str, float]]:
    """Compute mean/std for each metric across all images in *verify_IQA.json*."""
    with verify_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("Loaded verify_IQA.json with %d entries for aggregation", len(data))

    metrics: Dict[str, List[float]] = {}
    for img_scores in data.values():
        # Traverse nested dict to collect numeric values
        def _collect(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _collect(v, f"{prefix}.{k}" if prefix else k)
            elif isinstance(obj, (int, float)):
                metrics.setdefault(prefix, []).append(float(obj))
        _collect(img_scores)

    summary: Dict[str, Dict[str, float]] = {}
    for name, vals in metrics.items():
        if len(vals) < 2:
            summary[name] = {"mean": mean(vals), "std": 0.0}
        else:
            summary[name] = {"mean": mean(vals), "std": stdev(vals)}
    return summary


def main():
    logger.info("Starting experiments")
    experiment_names = load_experiment_names()

    # Load full base config once for template creation
    with BASE_YAML.open("r", encoding="utf-8") as fh:
        base_cfg = yaml.safe_load(fh)["experiments"]

    overall_collections: List[Dict[str, Dict[str, float]]] = []

    supported_exts = base_cfg["base"].get("general", {}).get("supported_extensions", ["jpg", "jpeg", "png"])

    for subfolder in sorted(D3_ROOT.iterdir()):
        if not subfolder.is_dir():
            continue
        logger.info("=== Processing sub-folder: %s ===", subfolder.name)

        # Collect images present at the root of subfolder to replicate for each experiment
        img_patterns = [f"*.{e}" for e in supported_exts]
        src_imgs = [p for pat in img_patterns for p in subfolder.glob(pat)]
        # If images already moved into a common raw directory, use them as source
        if not src_imgs:
            root_raw = subfolder / "raw"
            if root_raw.exists():
                src_imgs = list(root_raw.glob("*"))

        for exp_name in experiment_names:
            logger.info("Running experiment: %s", exp_name)

            exp_root = subfolder / exp_name
            raw_dir = exp_root / "raw"
            processed_dir = exp_root / "processed"
            artefacts_dir = exp_root / "artefacts"
            raw_dir.mkdir(parents=True, exist_ok=True)
            processed_dir.mkdir(exist_ok=True)
            artefacts_dir.mkdir(exist_ok=True)

            # Copy images to experiment raw directory if not already copied
            if not any(raw_dir.iterdir()):
                for img in src_imgs:
                    shutil.copy2(img, raw_dir / img.name)

            tmp_cfg = create_temp_config(base_cfg, exp_name, raw_dir, processed_dir, artefacts_dir)
            run_experiment(tmp_cfg, exp_name)

            # Aggregate metrics for this experiment
            verify_json = artefacts_dir / "verify_IQA.json"
            if verify_json.exists():
                summary = aggregate_verify_scores(verify_json)
                out_path = RESULTS_ROOT / f"{subfolder.name}__{exp_name}.json"
                with out_path.open("w", encoding="utf-8") as fh:
                    json.dump(summary, fh, indent=2)
                overall_collections.append(summary)
                logger.info("Saved summary statistics to %s", out_path)
            else:
                logger.error("%s not found; skipping summarisation.", verify_json)

    # Compute overall mean/std across sub-folder summaries (per metric)
    if overall_collections:
        combined: Dict[str, List[float]] = {}
        for summary in overall_collections:
            for metric, stats in summary.items():
                combined.setdefault(metric, []).append(stats["mean"])
        overall = {m: {"mean": mean(v), "std": (stdev(v) if len(v) > 1 else 0.0)} for m, v in combined.items()}
        overall_path = RESULTS_ROOT / "overall.json"
        with overall_path.open("w", encoding="utf-8") as fh:
            json.dump(overall, fh, indent=2)
        logger.info("Saved overall summary to %s", overall_path)


if __name__ == "__main__":
    main()
