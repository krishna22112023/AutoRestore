import cv2
from typing import Optional
import os
from time import time
from PIL import Image
import dashscope
import logging
import requests
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir("src"))
from dashscope import MultiModalConversation

from src.utils.image_processing import resize_image, encode_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QwenImageEdit:
    def __init__(self, host: str = "127.0.0.1", port: int = 5005, use_qwen_api: bool = True):
        if use_qwen_api:
            dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
        else:
            self.url = f"http://{host}:{port}/qwen_edit"
    
    def query_local(self, image_path: str, degradation_type: str, severity: str, inference_steps: int, true_cfg_scale: float, negative_prompt: Optional[str], output_path: str) -> float:
        resp = requests.post(self.url, json={"image_path":image_path, "degradation_type":degradation_type, 
        "severity":severity, "inference_steps":inference_steps, "true_cfg_scale":true_cfg_scale, "negative_prompt":negative_prompt, "output_path":output_path})
        resp.raise_for_status()
        duration = resp.json()["duration"]
        return duration

    def query_api(self, image_path: str, degradation_type: str, severity: str, output_path: str) -> str:

        prompt = open(f"{root}/src/prompts/qwen_preprocessor.md").read()
        formatted_prompt = prompt.format(degradation_type=degradation_type, severity=severity)

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": encode_file(image_path)},
                    {"text": formatted_prompt}
                ]
            }
        ]
        api_key = os.getenv("DASHSCOPE_API_KEY")
        logger.info("processing image using qwen-image-edit API")
        start_time = time()
        response = MultiModalConversation.call(
            api_key=api_key,
            model="qwen-image-edit",
            messages=messages,
            result_format='message',
            stream=False,
            negative_prompt=""
        )
        end_time = time()
        logger.info("processing done response: ", response)
        logger.info(f"inference time : {end_time - start_time} seconds")
        if response.status_code == 200:
            content = response.output["choices"][0]["message"]["content"][0]["image"]
            img_response = requests.get(content)
            image = cv2.imread(image_path)
            if image is not None:
                height, width, channels = image.shape
            else:
                with Image.open(image_path) as img:
                    width, height = img.size
            
            # Resize the processed image to match the original resolution using Torch with CUDA acceleration
            resized_pil = resize_image(img_response, height, width)
            save_path = os.path.join(
                output_path,
                f"{os.path.basename(image_path)}-{degradation_type}-{severity}.jpg"
            )
            resized_pil.save(save_path, format="JPEG")
            logger.info(f"Image resized and saved to {output_path}")
            return response
        else:
            logger.error(f"HTTP status code: {response.status_code}")
            logger.error(f"Error code: {response.code}")
            logger.error(f"Error message: {response.message}")
