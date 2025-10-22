import argparse
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

from tqdm import tqdm

# Add project root to PYTHONPATH so we can import the DepictQA wrapper
import pyprojroot, sys
root = pyprojroot.find_root(pyprojroot.has_dir("src"))
sys.path.append(str(root))

from src.autorestore.IQA import DepictQA as DepictQAWrapper  # noqa: E402

SEVERITY_LEVELS = ["very low", "low", "medium", "high", "very high"]
THRESHOLD_IDX = SEVERITY_LEVELS.index("medium")

DEGRADATIONS = [
    "motion blur",
    "defocus blur",
    "rain",
    "haze",
    "dark",
    "noise",
    "jpeg compression artifact",
    "low resolution",
]


def slugify(text: str) -> str:
    """Convert degradation text to a canonical form: lower-case, no spaces/underscores."""
    return text.replace(" ", "").replace("_", "").lower()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DepictQA degradation classification on a directory of images")
    parser.add_argument("--data_dir", type=Path, default=root / "eval/data/d3", help="Root directory containing degradation folders")
    parser.add_argument("--img_exts", type=str, nargs="*", default=[".png", ".jpg", ".jpeg", ".bmp"], help="Image extensions to consider")
    return parser.parse_args()


def collect_images(folder: Path, exts: List[str]) -> List[Path]:
    images: List[Path] = []
    for p in folder.iterdir():  # only direct children, no recursion
        if p.is_file() and p.suffix.lower() in exts:
            images.append(p)
    return images


def evaluate_folder(folder: Path, depictqa: DepictQAWrapper, img_exts: List[str]) -> Dict[str, float]:
    # Determine ground-truth degradations from folder name
    folder_slug = slugify(folder.name)
    gt_set: Set[str] = {d for d in DEGRADATIONS if slugify(d) in folder_slug}
    print(gt_set)
    if not gt_set:
        # Skip unknown folders gracefully
        return {}

    total_imgs = 0
    exact_match = 0
    TP = FP = FN = 0

    for img_path in tqdm(collect_images(folder, img_exts), desc=str(folder), leave=False):
        total_imgs += 1
        _, res = depictqa.query([img_path], task="eval_degradation")  # res: List[(degradation, severity)]
        pred_set = {d for d, sev in res if SEVERITY_LEVELS.index(sev) >= THRESHOLD_IDX}
        pred_set = {p for p in pred_set if p in DEGRADATIONS}  # ensure valid

        if pred_set == gt_set:
            exact_match += 1
        TP += len(pred_set & gt_set)
        FP += len(pred_set - gt_set)
        FN += len(gt_set - pred_set)

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = exact_match / total_imgs if total_imgs else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "images": total_imgs,
    }


def main():
    args = parse_args()
    data_dir: Path = args.data_dir
    assert data_dir.exists(), f"{data_dir} does not exist"

    depictqa = DepictQAWrapper(DEGRADATIONS, SEVERITY_LEVELS)

    folder_metrics: Dict[str, Dict[str, float]] = {}
    for folder in sorted([f for f in data_dir.iterdir() if f.is_dir()]):
        metrics = evaluate_folder(folder, depictqa, args.img_exts)
        if metrics:
            folder_metrics[folder.name] = metrics

    if not folder_metrics:
        print(f"No valid degradation folders found in {data_dir}.")
        return

    # Aggregate
    avg_metrics = {k: 0.0 for k in ["accuracy", "precision", "recall", "f1"]}
    for m in folder_metrics.values():
        for k in avg_metrics:
            avg_metrics[k] += m[k]
    num_folders = len(folder_metrics)
    if num_folders:
        for k in avg_metrics:
            avg_metrics[k] /= num_folders

    # Print report
    header = f"{'Folder':40s} |  Acc   Prec   Rec   F1   #Img"
    print(header)
    print("-" * len(header))
    for folder_name, m in folder_metrics.items():
        print(f"{folder_name:40s} | {m['accuracy']:.3f}  {m['precision']:.3f}  {m['recall']:.3f}  {m['f1']:.3f}  {int(m['images'])}")
    print("-" * len(header))
    print(f"{'Average':40s} | {avg_metrics['accuracy']:.3f}  {avg_metrics['precision']:.3f}  {avg_metrics['recall']:.3f}  {avg_metrics['f1']:.3f}")


if __name__ == "__main__":
    main()
