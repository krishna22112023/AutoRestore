import warnings
warnings.filterwarnings("ignore")
from typing import Optional
from pathlib import Path
from typing import Union
import os
import numpy as np
import torch
import requests
import pyprojroot
import sys
import cv2
from skimage.metrics import structural_similarity

root = pyprojroot.find_root(pyprojroot.has_dir("src"))
sys.path.append(str(root))

from config import logger
from src.utils.image_processing import read_image,resize_image

def ssim(raw_image: Path, proc_image: Path) -> float:
    """Compute Structural Similarity (SSIM) between *raw_image* and *proc_image*.

    Implementation follows the original SSIM formulation using a Gaussian
    window (11×11, σ=1.5) on each color channel and averages across channels.
    Returns the mean SSIM value in the range [-1, 1] where higher is better.
    """

    img1 = read_image(raw_image,normalize=True)
    img2 = read_image(proc_image,normalize=True)

    if img1.shape != img2.shape:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_CUBIC)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score = structural_similarity(gray1, gray2, data_range=img2.max() - img2.min())
    
    return float(score)


def psnr(raw_image: Path, proc_image: Path) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels between two images."""

    img1 = read_image(raw_image,normalize=True)
    img2 = read_image(proc_image,normalize=True)

    if img1.shape != img2.shape:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    score = cv2.PSNR(img1, img2)
    return score

class NR_IQA:
    def __init__(self, host: str = "127.0.0.1", port: int = 5003):
        self.url = f"http://{host}:{port}/iqa"

    def query(self, image_path: str, metric: str, proc_image_path: str = "") -> float:
        resp = requests.post(self.url, json={"raw_image_path": image_path, "proc_image_path": proc_image_path, "metric": metric}, timeout=30)
        resp.raise_for_status()
        return float(resp.json()["score"])

class ViT:
    def __init__(self, host: str = "127.0.0.1", port: int = 5004):
        self.url = f"http://{host}:{port}/vit_similarity"
    
    def query(self, raw_image: str, processed_image: str) -> float:
        resp = requests.post(self.url, json={"raw_image":raw_image, "processed_image":processed_image})
        resp.raise_for_status()
        return float(resp.json()["score"])

def restoration_score(raw_image: Path, proc_image: Path, _lambda: float, vit, qalign) -> float:
    #calculate scores
    if _lambda == 0.0:
        qalign_score = qalign.query(str(proc_image), "qalign")
        qalign_score_normalized = (qalign_score - 1) / 4
        return qalign_score_normalized
    else:
        if not os.path.exists(proc_image):
            logger.warning("Processed image not found for ViT query: %s", proc_image)
            return float("nan")
        cos_sim = vit.query(str(raw_image), str(proc_image))
        qalign_score = qalign.query(str(proc_image), "qalign")
        cos_sim_normalized = (cos_sim + 1) / 2
        qalign_score_normalized = (qalign_score - 1) / 4
        score = _lambda * cos_sim_normalized + (1-_lambda) * qalign_score_normalized
        return score

class DepictQA:
    """Parameters when called: img_path_lst, task (eval_degradation or comp_quality), degradations (if task is eval_degradation)."""
    def __init__(self,degradations, severity):
        self.degradations = degradations
        self.severity = severity

    def query(
        self,
        img_path_lst: list[Path],
        task: str,
        degradation: Optional[str] = None,
        replan: bool = False,
        previous_plan: Optional[str] = None
    ) -> tuple[str, str | list[tuple[str, str]]]:
        assert task in ["eval_degradation", "comp_quality"], f"Unexpected task: {task}"
        if task == "eval_degradation":
            assert (
                len(img_path_lst) == 1
            ), "Only one image should be provided for degradation evaluation."
            return self.eval_degradation(img_path_lst[0], degradation, replan, previous_plan)
        else:
            assert (
                len(img_path_lst) == 2
            ), "Two images should be provided quality comparison."
            return self.compare_img_qual(img_path_lst[0], img_path_lst[1])

    def eval_degradation(
        self, img: Path, degradation: Optional[str], replan: bool = False, previous_plan: Optional[str] = None
    ) -> tuple[str, list[tuple[str, str]]]:
        all_degradations: list[str] = self.degradations
        if degradation is None:
            degradations_lst = all_degradations
        else:
            if degradation == "low resolution":
                degradation = "low resolution"
            else:
                assert isinstance(
                    degradation, str
                ), f"Unexpected type of degradations: {type(degradation)}"
                assert (
                    degradation in all_degradations
                ), f"Unexpected degradation: {degradation}"
            degradations_lst = [degradation]

        levels: set[str] = self.severity
        res: list[tuple[str, str]] = []
        if replan:
            depictqa_evaluate_degradation_prompt = open(f"{root}/src/prompts/depictqa_eval_replan.md").read()
            logger.info(f"Re-evaluating degradations for {img.name} with prompt: {previous_plan}")
        else:
            depictqa_evaluate_degradation_prompt = open(f"{root}/src/prompts/depictqa_eval.md").read()
            logger.info(f"Evaluating degradation for {img.name}")
        for degradation in degradations_lst:
            if replan:
                prompt = depictqa_evaluate_degradation_prompt.format(
                    degradation=degradation, previous_plan=previous_plan
                )
            else:
                prompt = depictqa_evaluate_degradation_prompt.format(
                    degradation=degradation
                )
            url = "http://127.0.0.1:5001/evaluate_degradation"
            payload = {"imageA_path": img.resolve(), "prompt": prompt}
            rsp: str = requests.post(url, data=payload).json()["answer"]
            assert rsp in levels, f"Unexpected response from DepictQA: {list(rsp)}"
            res.append((degradation, rsp))

        if replan:
            prompt_to_display = depictqa_evaluate_degradation_prompt.format(
                degradation=degradations_lst, previous_plan=previous_plan
            )
        else:
            prompt_to_display = depictqa_evaluate_degradation_prompt.format(
                degradation=degradations_lst
            )
        return prompt_to_display, res

    def compare_img_qual(self, img1: Path, img2: Path) -> tuple[str, str]:
        prompt = open(f"{root}/src/prompts/depictqa_compare.md").read()
        url = "http://127.0.0.1:5002/compare_quality"
        payload = {
            "imageA_path": img1.resolve(),
            "imageB_path": img2.resolve(),
            "prompt": prompt
        }
        rsp: str = requests.post(url, data=payload).json()["answer"]

        if "A" in rsp and "B" not in rsp:
            choice = "former"
        elif "B" in rsp and "A" not in rsp:
            choice = "latter"
        else:
            raise ValueError(f"Unexpected answer from DepictQA: {rsp}")

        return prompt, choice

if __name__ == "__main__":
    from glob import glob
    dir = "/home/krishna/workspace/AutoRestore/eval/data/d3/dark_defocus_blur_jpeg compression_artifact"
    degradations = [
          "motion blur", "defocus blur", "rain",
          "haze", "dark", "noise", "jpeg compression artifact", "low resolution"
        ]
    severity = ["very low", "low", "medium", "high", "very high"]
    depictqa = DepictQA(degradations, severity)
    for image in glob(f"{dir}/*.png"):
        print(image)
        prompt, choice = depictqa.eval_degradation(Path(image), None, False, None)
        print(choice)
        break


    
