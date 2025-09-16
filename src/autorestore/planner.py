from __future__ import annotations

from pathlib import Path
from itertools import permutations
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json
import os
import random
import shutil
from typing import Dict, List, Tuple
import pyprojroot
import sys

root = pyprojroot.find_root(pyprojroot.has_dir("src"))
sys.path.append(str(root))

from src.autorestore.IQA import DepictQA, NR_IQA, ViT, restoration_score
from src.autorestore.preprocess import QwenImageEdit
from src.utils.file import ensure_dir, write_json, read_json
from config.run_config import load_config
from config import logger

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

    def __init__(self, config: dict) -> None:
        logger.info("Starting planning stage")
        # Load global configuration once
        planner_config = config.get("planner", {})
        iqa_config = planner_config.get("IQA", {})
        self.severity = iqa_config.get("severity", ["very low","low","medium", "high", "very high"])
        self.severity_threshold = iqa_config.get("severity_threshold", ["medium", "high", "very high"])
        self.degradations = iqa_config.get(
            "degradations",
            [
                "motion blur",
                "defocus blur",
                "rain",
                "haze",
                "dark",
                "noise",
                "jpeg compression artifact",
            ],
        )
        self.enable = planner_config.get("enable", True)
        self.max_degradations = iqa_config.get("max_degradations", 3)
        self._lambda = planner_config.get("score", {}).get("lambda", 0.5)
        self.optimize_pipeline = planner_config.get("optimize_pipeline", True)
        self.batch_by_degradation = planner_config.get("batch_by_degradation", True)
        logger.info(f"enable : {self.enable}, optimize_pipeline : {self.optimize_pipeline}, max_degradations : {self.max_degradations}, ")
        logger.info(f"Planning configuration: Detecting degradations : {self.degradations}, severity : {self.severity}, filter by severity {self.severity_threshold}")

        #initialize paths and models
        general_config = config.get("general", {})
        self.data_path = Path(root / general_config.get("data_path", "data/raw"))
        self.artefacts_path = Path(root / general_config.get("artefacts_path", "data/artefacts"))
        self.num_workers = general_config.get("num_workers", 4)
        ensure_dir(self.artefacts_path)
        self.depictqa = DepictQA(self.degradations, self.severity)
        self.qalign = NR_IQA()
        self.vit = ViT()    
        preprocess_config = config.get("executor", {}).get("preprocess", {})
        self.qwen = QwenImageEdit(use_api=preprocess_config.get("use_api", True),name=preprocess_config.get("name", "qwen-edit"))
        self.use_api = preprocess_config.get("use_api", True)
        
    def image_quality_analysis(
        self,
        images_subset: List[str] | None = None,
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
        logger.info(f"Found {len(image_paths_all)} images in {self.data_path}")
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
                if self.enable:
                    logger.info("Planning is enabled – running DepictQA.")
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
                else:
                    logger.info("Planning is disabled – randomly generating degradation-severity pairs.")
                    # Planning is disabled – randomly generate degradation-severity pairs
                    # Select up to *max_degradations* unique degradations from the configured list
                    deg_choices = random.sample(self.degradations, k=min(self.max_degradations, len(self.degradations)))
                    sev_list = list(self.severity)
                    res = [(d, random.choice(sev_list)) for d in deg_choices]
                logger.info(f"DepictQA result for {img_path.name}: {res}")

            except Exception as e:  
                logger.error("DepictQA failed for %s: %s", img_path.name, e)
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

        from itertools import combinations

        def _key_from_combo(combo: Tuple[str, ...]) -> str:
            return "-".join(sorted(d.replace(" ", "_") for d in combo))

        # Initialise dictionary with *all* possible combos
        batch_dict: Dict[str, List[str]] = {
            _key_from_combo(c): []
            for r in range(1, self.max_degradations + 1)
            for c in combinations(self.degradations, r)
        }

        severity_rank = {sev: i for i, sev in enumerate(reversed(self.severity_threshold))}

        for img_name, deg_list in per_image.items():
            # Filter degradations by severity ≥ medium
            valid: List[Tuple[str, str]] = [
                (d, s.lower()) for d, s in deg_list if s.lower() in severity_rank
            ]
            if not valid:
                continue  # skip images without significant degradation+++

            # Sort by severity rank desc, keep unique degradation names
            valid_sorted = sorted(valid, key=lambda x: -severity_rank[x[1]])
            unique_deg = []
            for d, _ in valid_sorted:
                if d not in unique_deg:
                    unique_deg.append(d)
                if len(unique_deg) == self.max_degradations:
                    break

            key = _key_from_combo(tuple(unique_deg))
            batch_dict[key].append(img_name)

        out_path = self.artefacts_path / "batch_IQA.json"
        write_json(batch_dict, out_path)
        logger.info("batch_IQA.json written to %s", out_path)

    def batch_process(
        self,
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

        num_workers = self.num_workers
        batch_file = self.artefacts_path / "batch_IQA.json"
        if not batch_file.exists():
            raise FileNotFoundError(f"{batch_file} not found. Run image_quality_analysis first.")

        # Load the mapping of degradation-combo → image list
        batches: Dict[str, List[str]] = read_json(batch_file)

        # Load any existing pipelines once, so it's available to both branches
        existing_pipelines: Dict[str, List[str]] = {}
        pipeline_file = self.artefacts_path / "batch_pipeline.json"
        if pipeline_file.exists():
            try:
                existing_pipelines = read_json(pipeline_file)
            except Exception:
                existing_pipelines = {}

        # If pipeline optimization is disabled, randomly assign an order for each degradation combo
        if not self.optimize_pipeline:
            logger.info("optimize_pipeline flag is False – generating random pipelines instead of exhaustive search.")
            best_pipelines: Dict[str, List[str]] = {}

            for degradation_id, _ in batches.items():
                # Reuse existing pipeline if we're not replanning and it exists already
                if (not replan) and (degradation_id in existing_pipelines):
                    best_pipelines[degradation_id] = existing_pipelines[degradation_id]
                    continue

                # Derive degradation list from the degradation_id key and shuffle it
                degradation_types = [d.replace("_", " ") for d in degradation_id.split("-")]
                random.shuffle(degradation_types)
                best_pipelines[degradation_id] = degradation_types

            # Merge with untouched existing pipelines and persist results
            final_pipelines = {**existing_pipelines, **best_pipelines}
            out_path = self.artefacts_path / "batch_pipeline.json"
            write_json(final_pipelines, out_path)
            logger.info("batch_pipeline.json written to %s", out_path)
            return  # Skip the optimization branch entirely

        # If pipeline optimization is enabled, perform exhaustive search
        logger.info("optimize_pipeline flag is True – performing exhaustive search.")

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
        img_deg_list: List[Tuple[str, str]]
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
                if self.use_api:
                    save_path = self.qwen.query_api_planner(
                        str(current_img_path), degradation, severity, str(tmp_out_dir)
                    )
                else:
                    # For local execution we fake a single-step pipeline containing the degradation
                    save_path = self.qwen.query_local(
                        str(current_img_path),
                        [{"degradation": degradation, "severity": severity}],
                        str(tmp_out_dir),
                    )

                # Fall back to deterministic name if backend didn't return a path
                if not save_path:
                    save_path = tmp_out_dir / f"{current_img_path.stem}-{degradation}-{severity}.jpg"
                current_img_path = Path(save_path)

            try:
                score = restoration_score(img_path, current_img_path, self._lambda, self.vit, self.qalign)
                logger.info("Permutation %s → score %.4f", "->".join(perm), score)
            except Exception as e:
                logger.error("Restoration score calculation failed. raw: %s, current: %s, lambda: %s", img_path, current_img_path, self._lambda)
                logger.exception("Error details: %s", e)
                score = -float("inf")
            if score > best_score:
                best_score = score
                best_sequence = list(perm)

        return best_sequence