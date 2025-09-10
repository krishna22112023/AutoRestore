import warnings
warnings.filterwarnings("ignore")
from typing import Optional
from pathlib import Path
from typing import Union
import numpy as np
import requests
import pyprojroot
import sys
import cv2

root = pyprojroot.find_root(pyprojroot.has_dir("src"))
sys.path.append(str(root))

from config import logger
from src.utils.image_processing import read_image

def ssim(raw_image: Path, proc_image: Path) -> float:
    """Compute Structural Similarity (SSIM) between *raw_image* and *proc_image*.

    Implementation follows the original SSIM formulation using a Gaussian
    window (11×11, σ=1.5) on each color channel and averages across channels.
    Returns the mean SSIM value in the range [-1, 1] where higher is better.
    """

    img1 = read_image(raw_image)
    img2 = read_image(proc_image)

    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape for SSIM computation")

    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    # Gaussian kernel used in the original SSIM paper
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    ssim_map = []
    for c in range(3):
        mu1 = cv2.filter2D(img1[:, :, c], -1, window)
        mu2 = cv2.filter2D(img2[:, :, c], -1, window)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.filter2D(img1[:, :, c] ** 2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(img2[:, :, c] ** 2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(img1[:, :, c] * img2[:, :, c], -1, window) - mu1_mu2

        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map.append(numerator / (denominator + 1e-12))

    ssim_val = np.mean(ssim_map)
    return float(ssim_val)


def psnr(raw_image: Path, proc_image: Path) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels between two images."""

    img1 = read_image(raw_image)
    img2 = read_image(proc_image)

    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape for PSNR computation")

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    return float(20 * np.log10(1.0 / np.sqrt(mse)))

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
    cos_sim = vit.query(str(raw_image), str(proc_image))
    qalign_score = qalign.query(str(proc_image), "qalign")
    #min max normalize score
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
                degradation = "blur"
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
    '''degradations = ["motion blur", "defocus blur", "rain", "haze", "dark", "noise", "jpeg compression artifact"]
    severity = {"very low", "low", "medium", "high", "very high"}
    depictqa = DepictQA(degradations, severity)
    print(depictqa.query([Path("/home/krishna/workspace/AutoRestore/data/raw/rain_storm-022.jpg")], "eval_degradation")) '''
    nr_iqa = NR_IQA()
    print(nr_iqa.query("/home/krishna/workspace/AutoRestore/data/raw/foggy-048.jpg", "qalign"))