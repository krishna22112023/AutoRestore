from __future__ import annotations

from pathlib import Path
import logging
import shutil
from typing import Dict, List, Tuple
import pyprojroot
import sys

root = pyprojroot.find_root(pyprojroot.has_dir("src"))
sys.path.append(str(root))

from src.autorestore.preprocess import QwenImageEdit
from src.utils.file import ensure_dir, read_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        data_path: str | Path,
        artefacts_path: str | Path,
        processed_path: str | Path,
        *,
        num_workers: int = 4,
        model_type: str = "qwen-edit",
        batch: bool = True,
        use_qwen_api: bool = True,
    ) -> None:
        root_path = root  # from earlier pyprojroot lookup
        self.data_path = Path(root_path / data_path)
        self.artefacts_path = Path(root_path / artefacts_path)
        self.processed_path = Path(root_path / processed_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        self.model_type = model_type
        self.batch = batch
        self.qwen = QwenImageEdit(use_qwenapi=use_qwen_api)
        self.use_qwen_api = use_qwen_api

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

            current_img_path = raw_path
            tmp_dir = tmp_root / img_name
            ensure_dir(tmp_dir)
            for step_idx, degradation in enumerate(seq):
                # seq items are space-separated original names
                sev = severity_map.get(degradation, "medium")
                logger.info(f"Step {step_idx+1} of {len(seq)}: Processing {img_name} with degradation {degradation} and severity {sev}")
                out_dir = tmp_dir
                if self.use_qwen_api:
                    self.qwen.query_api(str(current_img_path), degradation, sev, str(out_dir))
                else:
                    self.qwen.query_local(str(current_img_path), degradation, sev, 20, 9.0, " ", str(out_dir))
                # Determine saved image path
                current_img_path = out_dir / f"{current_img_path.name}-{degradation}-{sev}.jpg"

            # Copy / move final image to processed_path with original filename
            final_path = self.processed_path / img_name
            try:
                shutil.move(str(current_img_path), final_path)
            except shutil.Error:
                # Different filesystem – fallback to copy
                shutil.copy(str(current_img_path), final_path)

            # Cleanup tmp
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if self.batch:
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