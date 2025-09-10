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
from pathlib import Path

from src.utils.image_processing import resize_image, encode_file
from config import logger

class QwenImageEdit:
    """Wrapper around Qwen Image Edit API or local HF model.

    Parameters default to values specified in *config/base.yaml* under
    ``executor.preprocess`` but can still be overridden explicitly.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        use_api: bool | None = None,
        inference_steps: int | None = None,
        true_cfg_scale: float | None = None,
        name: str | None = None,
        default_size: tuple[int, int] | None = None,
        device_map: str | None = None,
    ):
        
        if use_api:
            dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
            self.name = name
        else:
            host = host or _prep_cfg.get("host", "127.0.0.1")
            port = port or int(_prep_cfg.get("port", 5005))
            self.url = f"http://{host}:{port}/qwen_edit"
            self.hf_name = name
            self.default_size = default_size
            self.device_map = device_map
            self.inference_steps = inference_steps
            self.true_cfg_scale = true_cfg_scale
    
    def query_local(self, image_path: str, pipeline: list[dict[str, str]], output_path: str) -> str:
        """Send the full degradation pipeline to the local Qwen server.

        Returns the filepath of the saved, processed image."""
        resp = requests.post(
            self.url,
            json={
                "image_path": image_path,
                "pipeline": pipeline,
                "hf_name": self.hf_name,
                "default_size": self.default_size,
                "device_map": self.device_map,
                "inference_steps": self.inference_steps,
                "true_cfg_scale": self.true_cfg_scale,
                "output_path": output_path,
            },
        )
        resp.raise_for_status()
        return resp.json().get("save_path", "")

    def query_api(self, image_path: str, pipeline: list[dict[str, str]], output_path: str) -> str:
        """Call DashScope multimodal API with full degradation pipeline.

        Returns the filepath of the saved, processed image."""

        prompt_template = open(f"{root}/src/prompts/qwen_preprocessor.md").read()
        formatted_prompt = prompt_template.replace("{degradation_type : severity}", str(pipeline))

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": encode_file(image_path)},
                    {"text": formatted_prompt}
                ]
            }
        ]
        logger.info("Processing image via Qwen API with full pipeline of %d steps", len(pipeline))
        start_time = time()
        response = MultiModalConversation.call(
            model=self.name,
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
                height, width = image.shape[:2]
            else:
                with Image.open(image_path) as img:
                    width, height = img.size

            resized_pil = resize_image(img_response, height, width)
            ext = Path(image_path).suffix or ".jpg"
            save_path = os.path.join(output_path, f"{Path(image_path).stem}-restored{ext}")
            resized_pil.save(save_path)
            logger.info("Image saved to %s", save_path)
            return save_path
        else:
            logger.error(f"HTTP status code: {response.status_code}")
            logger.error(f"Error code: {response.code}")
            logger.error(f"Error message: {response.message}")
            return ""
