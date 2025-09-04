from __future__ import annotations

from pathlib import Path
from itertools import permutations
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json
import logging
import os
import random
import shutil
from typing import Dict, List, Tuple
import pyprojroot
import sys

root = pyprojroot.find_root(pyprojroot.has_dir("src"))
sys.path.append(str(root))

from src.autorestore.IQA import DepictQA, QAlign, ViT 
from src.autorestore.preprocess import QwenImageEdit
from src.autorestore.metrics import restoration_score
from src.utils.file import ensure_dir, write_json, read_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Planning:
    """Implements the *Planning* stage of AutoRestore.

    Parameters
    ----------
    data_path : str | Path
        Directory that contains raw input images.
    artefacts_path : str | Path
        Directory where IQA/batch/pipeline JSON files and temporary images will
        be written.
    num_workers : int, default=4
        Degree of parallelism when running *qwen_preprocess*.
    """

    SEVERITY_ORDER = ["very low", "low", "medium", "high", "very high"]
    SEVERITY_THRESHOLD = {"medium", "high", "very high"}

    def __init__(self, data_path: str | Path, artefacts_path: str | Path, *, num_workers: int = 4, use_qwen_api: bool = True) -> None:
        self.data_path = Path(root / data_path)
        self.artefacts_path = Path(root / artefacts_path)
        self.num_workers = num_workers
        ensure_dir(self.artefacts_path)
        self.depictqa = DepictQA()
        self.qalign = QAlign()
        self.vit = ViT()    
        self.qwen = QwenImageEdit(use_qwenapi=use_qwen_api)
        self.use_qwen_api = use_qwen_api
        

    def image_quality_analysis(
        self,
        images_subset: List[str] | None = None,
        *,
        replan: bool = False,
        previous_plans: Dict[str, str] | None = None,
    ) -> None:
        """Run DepictQA on every image and persist full results to *IQA.json*.

        The resulting JSON maps *filename* → list[(degradation, severity)]. No
        filtering is applied at this stage.
        """
        logger.info("Running image-quality analysis with DepictQA…")

        ext = ("*.jpg", "*.jpeg", "*.png")
        image_paths_all = [p for pattern in ext for p in self.data_path.glob(pattern)]
        if images_subset is not None:
            names_set = set(images_subset)
            image_paths = [p for p in image_paths_all if p.name in names_set]
        else:
            image_paths = image_paths_all
        if not image_paths:
            logger.warning("No images found for IQA – aborting planning stage.")
            return

        per_image: Dict[str, List[Tuple[str, str]]] = {}

        for img_path in image_paths:
            try:
                prev_plan_str = None
                if replan and previous_plans is not None:
                    prev_plan_str = previous_plans.get(img_path.name)
                # DepictQA returns (prompt, list[(degradation, level), …])
                _, res = self.depictqa.query(
                    [img_path],
                    "eval_degradation",
                    replan=replan,
                    previous_plan=prev_plan_str,
                )
            except Exception as e:  
                logger.warning("DepictQA failed for %s: %s", img_path.name, e)
                res = []

            # Store full list – keep DepictQA wording intact
            per_image[img_path.name] = res

        if not per_image:
            logger.warning("No images with severity ≥ medium found – nothing to batch.")
            return

        # Persist IQA.json
        iqa_path = self.artefacts_path / "IQA.json"
        write_json(per_image, iqa_path)
        logger.info("IQA.json written to %s", iqa_path)

        # Derive batches next
        self.batch_creation(per_image)

    def batch_creation(self, per_image: Dict[str, List[Tuple[str, str]]]) -> None:
        """Generate *batch_IQA.json* where keys are unique degradation combos (≤3)."""

        all_degradations = [
            "motion blur",
            "defocus blur",
            "rain",
            "raindrop",
            "haze",
            "dark",
            "noise",
            "jpeg compression artifact",
        ]

        from itertools import combinations

        def _key_from_combo(combo: Tuple[str, ...]) -> str:
            return "-".join(sorted(d.replace(" ", "_") for d in combo))

        # Initialise dictionary with *all* possible combos (92 keys)
        batch_dict: Dict[str, List[str]] = {
            _key_from_combo(c): []
            for r in (1, 2, 3)
            for c in combinations(all_degradations, r)
        }

        severity_rank = {"very high": 3, "high": 2, "medium": 1}

        for img_name, deg_list in per_image.items():
            # Filter degradations by severity ≥ medium
            valid: List[Tuple[str, str]] = [
                (d, s.lower()) for d, s in deg_list if s.lower() in severity_rank
            ]
            if not valid:
                continue  # skip images without significant degradation

            # Sort by severity rank desc, keep unique degradation names
            valid_sorted = sorted(valid, key=lambda x: -severity_rank[x[1]])
            unique_deg = []
            for d, _ in valid_sorted:
                if d not in unique_deg:
                    unique_deg.append(d)
                if len(unique_deg) == 3:
                    break

            key = _key_from_combo(tuple(unique_deg))
            batch_dict[key].append(img_name)

        out_path = self.artefacts_path / "batch_IQA.json"
        write_json(batch_dict, out_path)
        logger.info("batch_IQA.json written to %s", out_path)

    def batch_process(
        self,
        *,
        num_workers: int | None = None,
        batch_process: bool = True,
        replan: bool = False,
    ) -> None:
        """Run pipeline search in parallel.

        Writes *batch_pipeline.json* containing the optimal sequence per
        degradation-ID.
        """
        if not batch_process:
            logger.info("batch_process flag is False – skipping exhaustive search.")
            return

        num_workers = num_workers or self.num_workers
        batch_file = self.artefacts_path / "batch_IQA.json"
        if not batch_file.exists():
            raise FileNotFoundError(f"{batch_file} not found. Run image_quality_analysis first.")

        # Load the mapping of degradation-combo → image list
        batches: Dict[str, List[str]] = read_json(batch_file)

        pipeline_file = self.artefacts_path / "batch_pipeline.json"
        existing_pipelines: Dict[str, List[str]] = {}
        if pipeline_file.exists():
            try:
                existing_pipelines = read_json(pipeline_file)
            except Exception as e:
                logger.warning("No existing pipeline file – proceeding with empty set: %s", e)
                existing_pipelines = {}

        # Load IQA for severity lookup
        iqa_data: Dict[str, List[Tuple[str, str]]] = read_json(self.artefacts_path / "IQA.json")

        tmp_dir = self.artefacts_path / "tmp_planning"
        ensure_dir(tmp_dir)

        best_pipelines: Dict[str, List[str]] = {}

        # Build an iterator over batches in chunks of *num_workers*
        batch_items = list(batches.items())
        for i in range(0, len(batch_items), num_workers):
            chunk = batch_items[i : i + num_workers]
            futures = {}
            with ThreadPoolExecutor(max_workers=num_workers) as exe:
                for degradation_id, img_list in chunk:
                    # If we already have a pipeline and we're *not* replanning, reuse it and skip exhaustive search
                    if (not replan) and (degradation_id in existing_pipelines):
                        best_pipelines[degradation_id] = existing_pipelines[degradation_id]
                        logger.info(f"Reused pipeline for {degradation_id}. Skipping exhaustive search.")
                        continue

                    # Otherwise, perform exhaustive search
                    all_imgs = img_list
                    if not all_imgs:
                        continue
                    img_choice = Path(self.data_path / random.choice(all_imgs))
                    futures[
                        exe.submit(
                            self._search_best_pipeline,
                            degradation_id,
                            img_choice,
                            iqa_data[img_choice.name],
                        )
                    ] = degradation_id

                for fut in as_completed(futures):
                    degradation_id = futures[fut]
                    try:
                        best_seq = fut.result()
                        if best_seq:
                            best_pipelines[degradation_id] = best_seq
                    except Exception as e:
                        logger.error("Pipeline search failed for %s: %s", degradation_id, e)

        # Merge with any untouched existing pipelines so nothing gets lost
        final_pipelines = {**existing_pipelines, **best_pipelines}

        out_path = self.artefacts_path / "batch_pipeline.json"
        write_json(final_pipelines, out_path)
        logger.info("batch_pipeline.json written to %s", out_path)

        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def _search_best_pipeline(
        self,
        degradation_id: str,
        img_path: Path,
        img_deg_list: List[Tuple[str, str]],
        gamma: float = 0.5,
    ) -> List[str] | None:
        """Enumerate all permutations of the degradation list and return the best sequence."""
        degradation_types = [d.replace("_", " ") for d in degradation_id.split("-")]
        best_score = float("-inf")
        best_sequence: List[str] | None = None

        # Build severity map from the chosen image's IQA list
        severity_map = {d: s for d, s in img_deg_list if d in degradation_types}
        severity_map = {d: severity_map.get(d, "medium") for d in degradation_types}

        for perm in permutations(degradation_types, len(degradation_types)):
            current_img_path = img_path
            for step_idx, degradation in enumerate(perm):
                severity = severity_map[degradation]
                tmp_out_dir = self.artefacts_path / "tmp_planning" / degradation_id / "_".join(perm)
                ensure_dir(tmp_out_dir)
                if self.use_qwen_api:
                    self.qwen.query_api(str(current_img_path), degradation, severity, str(tmp_out_dir))
                else:
                    self.qwen.query_local(str(current_img_path), degradation, severity, 20, 9.0, " ", str(tmp_out_dir))
                next_img = tmp_out_dir / f"{current_img_path.name}-{degradation}-{severity}.jpg"
                current_img_path = next_img

            # Finished the pipeline – compute quality score
            try:
                score = restoration_score(img_path, current_img_path, self.vit, self.qalign, gamma)
            except Exception:
                score = -float("inf")
            if score > best_score:
                best_score = score
                best_sequence = list(perm)

        return best_sequence