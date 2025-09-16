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

def _empty_flags(replan: bool = False) -> Dict[str, bool]:
    """Return a fresh dict containing all step flags set to *False*."""
    return {
        "planner.batching": False,
        "planner.optimization": False,
        "executor": False,
        "verifier": False,
        "replan": replan,
    }

def current_attempt_label(attempts_dict: Dict[str, Dict[str, bool]]) -> str:
    """Return the label (e.g. *attempt3*) of the *latest* attempt for an image."""
    return max(attempts_dict.keys(), key=lambda k: int(k.replace("attempt", "")))

def ensure_active_attempts(state: Dict[str, Dict[str, Dict[str, bool]]]) -> int:
    """Ensure there's an *active* attempt entry at the top level.

    If **any** image in the latest attempt finished the full pipeline *and* has
    ``replan=True``, we open **one** new attempt (``attemptN+1``) that contains
    *all* images (initialised with fresh step flags).

    The function mutates *state* in-place and returns the *current* attempt
    number (int).
    """
    latest_lbl = current_attempt_label(state)
    latest_attempt = state[latest_lbl]

    need_new_attempt = any(
        flags["replan"] and all(
            flags[k] for k in ("planner.batching", "planner.optimization", "executor", "verifier")
        )
        for flags in latest_attempt.values()
    )

    if need_new_attempt:
        next_idx = int(latest_lbl.replace("attempt", "")) + 1
        next_lbl = f"attempt{next_idx}"
        state[next_lbl] = {img: _empty_flags(replan=False) for img in latest_attempt.keys()}
        return next_idx
    return int(latest_lbl.replace("attempt", ""))


def images_to_process(state: Dict[str, Dict[str, Dict[str, bool]]]) -> List[str]:
    """Return images whose *verifier* flag is *False* in *current* attempt."""
    current_lbl = current_attempt_label(state)
    return [img for img, flags in state[current_lbl].items() if not flags["verifier"]]


def main(args):
    cfg = load_config(args.config_path, args.experiment)

    # General settings
    general_config = cfg.get("general", {})
    max_retries = int(general_config.get("max_retries", 3))
    artefacts_path = Path(root / general_config.get("artefacts_path", "data/artefacts"))
    data_path = Path(root / general_config.get("data_path", "data/raw"))
    supported_exts = general_config.get("supported_extensions", ["jpg", "jpeg", "png"])
    attempt = 0

    #check if api is loaded
    if cfg.get("executor", {}).get("preprocess", {}).get("use_api"):
        if os.environ.get("DASHSCOPE_API_KEY") is None:
            logger.error("DASHSCOPE_API_KEY is not set. Please create a .env file in the root directory and set the DASHSCOPE_API_KEY")
            raise ValueError("DASHSCOPE_API_KEY is not set")
        else:
            logger.info("DASHSCOPE_API_KEY is set")

    state_path = artefacts_path / "state.json"

    # If *state.json* does not yet exist, create it with *attempt1* entries
    if not state_path.exists():
        logger.info("Creating initial state")
        all_imgs = [p.name for ext in supported_exts for p in data_path.glob(f"*.{ext}")]
        initial_state: Dict[str, Dict[str, Dict[str, bool]]] = {
            "attempt1": {img: _empty_flags(replan=False) for img in all_imgs}
        }
        write_json(initial_state, state_path)
        attempt = 1
        state = read_json(state_path)
    else:
        logger.info("Loading existing state")
        state = read_json(state_path)
        attempt = ensure_active_attempts(state)
        write_json(state, state_path)
    for att in range(attempt, max_retries + 1):
        state = read_json(state_path)
        to_process: List[str] = images_to_process(state)
        if not to_process:
            logger.info("All images successfully restored after %d iteration(s).", att - 1)
            break

        logger.info("--- Iteration %d: processing %d image(s) ---", att, len(to_process))

        # Prepare previous plan mapping (if not first iteration)
        prev_plans_map: Dict[str, str] | None = None
        if att > 1:
            try:
                IQA_data = read_json(artefacts_path / "IQA.json")
                prev_plans_map = {key: IQA_data[key] for key in to_process if key in IQA_data}
                logger.info("previous IQA results loaded")
            except Exception:
                prev_plans_map = None
                logger.error("failed to load previous IQA results")
        # Planning restricted to failed/raw list of images
        planning = Planning(cfg)
        t0 = time.perf_counter()
        # --------------------------- PLANNER (BATCHING) --------------------
        current_lbl = current_attempt_label(state)
        imgs_needing_batching = [img for img in to_process if not state[current_lbl][img]["planner.batching"]]

        #image quality analysis
        t1 = time.perf_counter()
        if imgs_needing_batching:
            planning.image_quality_analysis(
                images_subset=imgs_needing_batching,
                replan=(att > 1),
                previous_plans=prev_plans_map,
            )

            for img in imgs_needing_batching:
                state[current_lbl][img]["planner.batching"] = True
            write_json(state, state_path)
        else:
            logger.info("Planning skipped. No images needing batching")
        iqa_time = time.perf_counter() - t1
        #pipeline optimization
        t2 = time.perf_counter()
        imgs_needing_opt = [img for img in to_process if not state[current_lbl][img]["planner.optimization"]]
        if imgs_needing_opt:
            planning.batch_process(replan=(att > 1))
            for img in imgs_needing_opt:
                state[current_lbl][img]["planner.optimization"] = True
            write_json(state, state_path)
        else:
            logger.info("Pipeline optimization skipped. No images needing optimization")
        bp_time = time.perf_counter() - t2

        # Execute
        t3 = time.perf_counter()
        executor = Executor(cfg)
        imgs_needing_exec = [img for img in to_process if not state[current_lbl][img]["executor"]]
        if imgs_needing_exec:
            executor.run()
            for img in imgs_needing_exec:
                state[current_lbl][img]["executor"] = True
            write_json(state, state_path)
        else:
            logger.info("Execution stage skipped. No images needing execution")
        exec_time = time.perf_counter() - t3

        # Verify
        t4 = time.perf_counter()
        verifier = Verifier(cfg)
        imgs_needing_ver = [img for img in to_process if not state[current_lbl][img]["verifier"]]
        if imgs_needing_ver:
            failed_imgs = verifier.run()
        else:
            logger.info("Verification stage skipped. No images needing verification")
        ver_time = time.perf_counter() - t4        

        # Inject speed entry into verify_IQA.json
        inference_speed_path = artefacts_path / "inference_speed.json"
        if inference_speed_path.exists():
            try:
                inference_speed_data = read_json(inference_speed_path)
                speed_entry: Dict[str, Dict | float] = inference_speed_data.get(f"attempt{att}", {})

                if imgs_needing_batching:
                    speed_entry["planner.batching"] = round(iqa_time, 2)
                if imgs_needing_opt:
                    speed_entry["planner.optimization"] = round(bp_time, 2)
                if imgs_needing_exec:
                    speed_entry["executor"] = round(exec_time, 2)
                if imgs_needing_ver:
                    speed_entry["verifier"] = round(ver_time, 2)
                write_json(inference_speed_data, inference_speed_path)
            except Exception as e:
                logger.error("Failed to write inference speed to %s: %s", inference_speed_path, e)

        # Update state based on verification outcome
        for img in imgs_needing_ver:
            lbl = current_lbl
            if img not in failed_imgs:
                # Success – mark verifier true, keep replan false
                state[lbl][img]["verifier"] = True
                state[lbl][img]["replan"] = False
            else:
                # Verification failed – mark verifier done but set replan true
                state[lbl][img]["verifier"] = True
                state[lbl][img]["replan"] = True
        # ensure_active_attempts will create a fresh attempt when needed in the next loop

        write_json(state, state_path)

    else:
        logger.info("Maximum retries (%d) reached. Some images still failed.", max_retries)

if __name__ == "__main__":
    # Load experiment configuration – default to 'base' or set via ENV EXPERIMENT
    parser = argparse.ArgumentParser(description="AutoRestore image restoration pipeline")
    parser.add_argument("--experiment", default="base", help="Experiment configuration to use (default: base)")
    parser.add_argument("--config_path", default="config/base.yaml", help="Path to yaml configuration file")
    args = parser.parse_args()
    main(args)