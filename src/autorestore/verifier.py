from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import pyprojroot
import sys
root = pyprojroot.find_root(pyprojroot.has_dir("src"))
sys.path.append(str(root))

from src.utils.file import ensure_dir, write_json
from src.autorestore.IQA import QAlign
from config import logger, settings

class Verifier:
    """Compare raw vs processed images using QAlign."""

    def __init__(self,config: dict,) -> None:
        logger.info("Starting verification stage")
        verifier_config = config.get("verifier", {})
        self.enable = verifier_config.get("enable", True)
        if not self.enable:
            logger.info("Verification stage disabled. verification skipped.")
            return
        general_config = config.get("general", {})
        self.data_path = Path(settings.BASE / general_config.get("data_path", "data/raw"))
        self.processed_path = Path(settings.BASE / general_config.get("processed_path", "data/processed"))
        self.artefacts_path = Path(settings.BASE / general_config.get("artefacts_path", "data/artefacts"))
        ensure_dir(self.artefacts_path)
        self.qalign = QAlign()

    def run(self) -> None:
        failed: List[str] = []
        verify_scores: Dict[str, Tuple[List, List]] = {}

        for proc_img in self.processed_path.glob("*.jpg"):
            logger.info(f"Verifying quality of {proc_img.name}")
            raw_img = self.data_path / proc_img.name
            if not raw_img.exists():
                logger.warning("Missing raw counterpart for %s", proc_img.name)
                continue

            try:
                # Individual degradation evaluation for record
                raw_score = self.qalign.query(str(raw_img))
                proc_score = self.qalign.query(str(proc_img))
                verify_scores[proc_img.name] = [raw_score, proc_score]
                logger.info(f"verified {proc_img.name} = raw image score: {raw_score}, processed image score: {proc_score}")
                if proc_score <= raw_score:
                    logger.info(f"verification failed for {proc_img.name}. appended to failed list to replan.")
                    failed.append(proc_img.name)
            except Exception as e:
                logger.error("Internal failure during verification %s: %s", proc_img.name, e)
            logger.info(f"verification passed for {proc_img.name}")

        # Save failed list
        failed_path = self.artefacts_path / "failed_IQA.json"
        write_json(failed, failed_path)
        logger.info("Verification complete. %d failures saved to %s", len(failed), failed_path)

        # Save evaluation details
        eval_path = self.artefacts_path / "verify_IQA.json"
        write_json(verify_scores, eval_path)
        logger.info("Per-image evaluation saved to %s", eval_path)