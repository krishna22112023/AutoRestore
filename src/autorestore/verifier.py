from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import pyprojroot
import sys
root = pyprojroot.find_root(pyprojroot.has_dir("src"))
sys.path.append(str(root))

from src.utils.file import ensure_dir, write_json
from src.autorestore.IQA import NR_IQA,psnr,ssim
from config import logger, settings

class Verifier:
    """Compare raw vs processed images using NR_IQA."""

    def __init__(self,config: dict,) -> None:
        logger.info("Starting verification stage")
        verifier_config = config.get("verifier", {})
        self.enable = verifier_config.get("enable", True)
        if not self.enable:
            logger.info("Verification stage disabled. verification skipped.")
            return
        general_config = config.get("general", {})
        self.supported_exts = general_config.get("supported_extensions", ["jpg", "jpeg", "png"]) 
        self.data_path = Path(settings.BASE / general_config.get("data_path", "data/raw"))
        self.processed_path = Path(settings.BASE / general_config.get("processed_path", "data/processed"))
        self.artefacts_path = Path(settings.BASE / general_config.get("artefacts_path", "data/artefacts"))
        ensure_dir(self.artefacts_path)
        self.nr_iqa = NR_IQA()

    def run(self) -> List[str]:
        failed: List[str] = []
        verify_scores: Dict[str, Dict[str, float]] = {}
        patterns = [f"*.{e}" for e in self.supported_exts]
        imgs = [p for pat in patterns for p in self.processed_path.glob(pat)]
        for proc_img in imgs:
            logger.info(f"Verifying quality of {proc_img.name}")
            raw_img = self.data_path / proc_img.name
            if not raw_img.exists():
                logger.warning("Missing raw counterpart for %s", proc_img.name)
                continue

            try:
                # no reference iqa metrics
                qalign_raw_score = self.nr_iqa.query(str(raw_img), "qalign")
                qalign_proc_score = self.nr_iqa.query(str(proc_img), "qalign")
                maniqa_raw_score = self.nr_iqa.query(str(raw_img), "maniqa")
                maniqa_proc_score = self.nr_iqa.query(str(proc_img), "maniqa")
                clipqa_raw_score = self.nr_iqa.query(str(raw_img), "clipiqa")
                clipqa_proc_score = self.nr_iqa.query(str(proc_img), "clipiqa")
                musiq_raw_score = self.nr_iqa.query(str(raw_img), "musiq")
                musiq_proc_score = self.nr_iqa.query(str(proc_img), "musiq")

                #full reference iqa metrics
                psnr_score = psnr(str(raw_img), str(proc_img))
                ssim_score = ssim(str(raw_img), str(proc_img))
                lpips_score = self.nr_iqa.query(str(raw_img), "lpips",str(proc_img))

                verify_scores[proc_img.name] = {"NR_IQA":{"qalign": {"raw": qalign_raw_score, "proc": qalign_proc_score},
                "maniqa": {"raw": maniqa_raw_score, "proc": maniqa_proc_score},
                "clipiqa": {"raw": clipqa_raw_score, "proc": clipqa_proc_score},
                "musiq": {"raw": musiq_raw_score, "proc": musiq_proc_score}}, 
                "FR_IQA":{"lpips": lpips_score, "psnr": psnr_score, "ssim": ssim_score}}

                logger.info(f"verified {proc_img.name} = raw image score: {qalign_raw_score}, processed image score: {qalign_proc_score}")
                if qalign_proc_score <= qalign_raw_score:
                    logger.info(f"verification failed for {proc_img.name}. appended to failed list to replan.")
                    failed.append(proc_img.name)
                
            except Exception as e:
                logger.error("Internal failure during verification %s: %s", proc_img.name, e)
            logger.info(f"verification passed for {proc_img.name}")

        # Save evaluation details
        eval_path = self.artefacts_path / "verify_IQA.json"
        write_json(verify_scores, eval_path)
        logger.info("Per-image evaluation saved to %s", eval_path)

        return failed