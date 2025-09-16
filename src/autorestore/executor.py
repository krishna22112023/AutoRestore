from __future__ import annotations

from pathlib import Path
import shutil
from typing import Dict, List, Tuple
import pyprojroot
import sys
import os

root = pyprojroot.find_root(pyprojroot.has_dir("src"))
sys.path.append(str(root))

from src.autorestore.preprocess import QwenImageEdit
from src.utils.file import ensure_dir, read_json
from config.run_config import load_config
from config import logger

class Executor:
    """Apply the optimal pipelines discovered by :class:`Planning`.

    Parameters
    ----------
    data_path : str | Path
        Directory containing raw images.
    artefacts_path : str | Path
        Directory where *batch_pipeline.json*, *batch_IQA.json* and *IQA.json*
        reside.
    processed_path : str | Path
        Destination directory for final, restored images.
    num_workers : int, default=4
        Parallel workers used when invoking *qwen_preprocess*.
    model_type : str, default="qwen-edit"
        Placeholder for future extensibility – currently unused.
    """

    def __init__(
        self,
        config: dict) -> None:
        logger.info("Starting execution stage")
        general_config = config.get("general", {})
        self.data_path = Path(root / general_config.get("data_path", "data/raw"))
        self.artefacts_path = Path(root / general_config.get("artefacts_path", "data/artefacts"))
        self.processed_path = Path(root / general_config.get("processed_path", "data/processed"))
        self.processed_path.mkdir(parents=True, exist_ok=True)

        executor_config = config.get("executor", {})
        preprocess_config = executor_config.get("preprocess", {})
        self.num_workers = int(config.get("general", {}).get("num_workers", 4))
        self.batch = bool(executor_config.get("batch", False))
        self.use_api = bool(preprocess_config.get("use_api", True))
        self.model_type = preprocess_config.get("name", "qwen-edit")

        if not self.use_api:
            self.hf_name = preprocess_config.get("hf_name", "Qwen/Qwen-Image-Edit")
            self.default_size = preprocess_config.get("default_size", (1664, 928))
            self.device_map = preprocess_config.get("device_map", "balanced")
            self.inference_steps = preprocess_config.get("inference_steps", 20)
            self.true_cfg_scale = preprocess_config.get("true_cfg_scale", 9.0)
            self.qwen = QwenImageEdit(use_api=self.use_api, hf_name=self.hf_name, default_size=self.default_size, device_map=self.device_map, inference_steps=self.inference_steps, true_cfg_scale=self.true_cfg_scale)
            logger.info(f"using local model {self.hf_name}.")
            logger.info("This will be slow. For <60s inference speed, please switch to API by setting use_api=True")
        else:
            self.qwen = QwenImageEdit(use_api=self.use_api, name=self.model_type)
            logger.info(f"using api model {self.model_type}")

        # Load artefacts
        self.batch_pipeline: Dict[str, List[str]] = read_json(self.artefacts_path / "batch_pipeline.json")
        self.batch_iqa: Dict[str, List[str]] = read_json(self.artefacts_path / "batch_IQA.json")
        self.iqa_data: Dict[str, List[Tuple[str, str]]] = read_json(self.artefacts_path / "IQA.json")

    def run(self) -> None:
        """Execute restoration pipelines in parallel."""
        tmp_root = self.artefacts_path / "tmp_execute"
        ensure_dir(tmp_root)

        tasks = []  # list of (img_name, degradation_seq)
        for combo_key, seq in self.batch_pipeline.items():
            imgs = self.batch_iqa.get(combo_key, [])
            if not imgs:
                continue
            for img_name in imgs:
                tasks.append((img_name, seq))

        if not tasks:
            logger.warning("Executor found no images to process.")
            return

        def _process(item):
            img_name, seq = item
            raw_path = self.data_path / img_name
            if not raw_path.exists():
                logger.error("Raw image missing: %s", raw_path)
                return

            # Build severity map for this image
            deg_list = self.iqa_data.get(img_name, [])
            severity_map = {d: s for d, s in deg_list}

            tmp_dir = tmp_root / img_name
            ensure_dir(tmp_dir)

            # Build full degradation pipeline list
            pipeline = [{deg: severity_map.get(deg, "medium")} for deg in seq]

            if self.use_api:
                saved_path = self.qwen.query_api_executor(str(raw_path), pipeline, str(tmp_dir))
            else:
                saved_path = self.qwen.query_local(str(raw_path), pipeline, str(tmp_dir))

            if not saved_path:
                logger.error("Processing failed (no output) for %s", img_name)
                return

            final_path_src = Path(saved_path)
            final_path = self.processed_path / img_name
            try:
                shutil.move(str(final_path_src), final_path)
                logger.info(f"Preprocessed image saved to {final_path}")
            except shutil.Error:
                # Different filesystem – fallback to copy
                shutil.copy(str(final_path_src), final_path)

            shutil.rmtree(tmp_dir, ignore_errors=True)

        if self.batch:
            # this will not work for api model since api doesn't support batching
            logger.warning("Batch processing is not supported for API model. Please set batch=False")
            return
            # Parallel execution
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=self.num_workers) as exe:
                futures = {exe.submit(_process, item): item for item in tasks}
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc:
                        img_name, _ = futures[fut]
                        logger.error("Processing failed for %s: %s", img_name, exc)
        else:
            for item in tasks:
                try:
                    _process(item)
                except Exception as e:
                    img_name, _ = item
                    logger.error("Processing failed for %s: %s", img_name, e)

        # Cleanup root tmp dir
        shutil.rmtree(tmp_root, ignore_errors=True)