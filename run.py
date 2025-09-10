from pathlib import Path
from typing import List, Dict
import pyprojroot
import time
import sys
import os
import argparse
from dotenv import load_dotenv
import dashscope

root = pyprojroot.find_root(pyprojroot.has_dir("src"))

sys.path.append(str(root))

from config.run_config import load_config
from config import logger, settings
from src.utils.file import write_json, read_json
from src.autorestore.planner import Planning
from src.autorestore.executor import Executor
from src.autorestore.verifier import Verifier

load_dotenv(Path(root, ".env"))
dashscope.api_key = settings.DASHSCOPE_API_KEY

def main(args):
    cfg = load_config(args.config_path, args.experiment)

    # General settings
    general_config = cfg.get("general", {})
    max_retries = int(general_config.get("max_retries", 3))
    artefacts_path = Path(root / general_config.get("artefacts_path", "data/artefacts"))
    data_path = Path(root / general_config.get("data_path", "data/raw"))
    supported_exts = general_config.get("supported_extensions", ["jpg", "jpeg", "png"])

    #check if api is loaded
    if cfg.get("executor", {}).get("preprocess", {}).get("use_api"):
        if os.environ.get("DASHSCOPE_API_KEY") is None:
            logger.error("DASHSCOPE_API_KEY is not set. Please create a .env file in the root directory and set the DASHSCOPE_API_KEY")
            raise ValueError("DASHSCOPE_API_KEY is not set")
        else:
            logger.info("DASHSCOPE_API_KEY is set")

    # Initialise raw.json if not present
    raw_list_path = artefacts_path / "state.json"
    if not raw_list_path.exists():
        all_raw = [p.name for ext in supported_exts for p in data_path.glob(f"*.{ext}")]
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

        # Planning restricted to failed/raw list of images
        planning = Planning(cfg)
        t0 = time.perf_counter()
        planning.image_quality_analysis(images_subset=to_process, replan=(attempt > 1), previous_plans=prev_plans_map)
        iq_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        planning.batch_process(replan=(attempt > 1))
        bp_time = time.perf_counter() - t1

        # Execute
        executor = Executor(cfg)
        t2 = time.perf_counter()
        executor.run()
        exec_time = time.perf_counter() - t2

        # Verify
        verifier = Verifier(cfg)
        t3 = time.perf_counter()
        verifier.run()
        ver_time = time.perf_counter() - t3

        # Assemble inference speed breakdown for this attempt
        speed_entry = {
            "planner": {
                "image_quality_analysis": round(iq_time, 2),
                "batch_process": round(bp_time, 2),
            },
            "executor": round(exec_time, 2),
            "verifier": round(ver_time, 2),
        }

        # Inject speed entry into verify_IQA.json
        verify_path = artefacts_path / "verify_IQA.json"
        if verify_path.exists():
            try:
                verify_data = read_json(verify_path)
                for img_meta in verify_data.values():
                    img_meta.setdefault("inference_speed", {})[f"attempt{attempt}"] = speed_entry
                write_json(verify_data, verify_path)
            except Exception as e:
                logger.error("Failed to write inference speed to %s: %s", verify_path, e)

        failed_path = artefacts_path / "failed_IQA.json"
        failed_imgs: List[str] = read_json(failed_path) if failed_path.exists() else []

        write_json(failed_imgs, raw_list_path)

    else:
        logger.info("Maximum retries (%d) reached. Some images still failed.", max_retries)

if __name__ == "__main__":
    # Load experiment configuration – default to 'base' or set via ENV EXPERIMENT
    parser = argparse.ArgumentParser(description="AutoRestore image restoration pipeline")
    parser.add_argument("--experiment", default="base", help="Experiment configuration to use (default: base)")
    parser.add_argument("--config_path", default="config/base.yaml", help="Path to yaml configuration file")
    args = parser.parse_args()
    main(args)