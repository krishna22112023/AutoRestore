from pathlib import Path
from typing import List, Dict
import pyprojroot
import sys
import logging
import os
from dotenv import load_dotenv

root = pyprojroot.find_root(pyprojroot.has_dir("src"))
sys.path.append(str(root))
load_dotenv(os.path.join(root,".env"))

from src.utils.file import write_json, read_json
from src.autorestore.planner import Planning
from src.autorestore.executor import Executor
from src.autorestore.verifier import Verifier

logger = logging.getLogger(__name__)

data_path = Path("data/raw")
artefacts_path = Path("data/artefacts")
processed_path = Path("data/processed")
max_retries = 3

# Initialise raw.json if not present
raw_list_path = artefacts_path / "raw.json"
if not raw_list_path.exists():
    all_raw = [p.name for p in data_path.glob("*.jpg")]
    write_json(all_raw, raw_list_path)

for attempt in range(1, max_retries + 1):
    to_process: List[str] = read_json(raw_list_path)
    if not to_process:
        logger.info("All images successfully restored after %d iteration(s).", attempt - 1)
        break

    logger.info("--- Iteration %d: processing %d image(s) ---", attempt, len(to_process))

    # Prepare previous plan mapping (if not first iteration)
    prev_plans_map: Dict[str, str] | None = None
    if attempt > 1:
        try:
            combo_to_plan = read_json(artefacts_path / "batch_pipeline.json")
            combo_to_imgs = read_json(artefacts_path / "batch_IQA.json")
            prev_plans_map = {}
            for combo, imgs in combo_to_imgs.items():
                if combo in combo_to_plan:
                    seq = combo_to_plan[combo]
                    seq_str = " -> ".join(seq)
                    for img in imgs:
                        prev_plans_map[img] = seq_str
        except Exception:
            prev_plans_map = None

    # Planning restricted to failed/raw list
    planning = Planning(data_path, artefacts_path)
    planning.image_quality_analysis(images_subset=to_process, replan=(attempt > 1), previous_plans=prev_plans_map)
    planning.batch_process(replan=(attempt > 1))

    # Execute
    executor = Executor(data_path, artefacts_path, processed_path, batch=False)
    executor.run()

    # Verify
    verifier = Verifier(data_path, processed_path, artefacts_path)
    verifier.run()

    failed_path = artefacts_path / "failed_IQA.json"
    failed_imgs: List[str] = read_json(failed_path) if failed_path.exists() else []

    write_json(failed_imgs, raw_list_path)

else:
    logger.info("Maximum retries (%d) reached. Some images still failed.", max_retries)