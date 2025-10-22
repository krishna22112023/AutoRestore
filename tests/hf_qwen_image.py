import os
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline
from src.autorestore.IQA import NR_IQA
from glob import glob

model = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16, device_map="balanced",device_map_kwargs={"num_devices": torch.cuda.device_count()})
print("model loaded")

pipeline = {"haze": "high","blur":"high"}
prompt_template = open("/home/krishna/workspace/AutoRestore/src/prompts/qwen_preprocessor_executor.md").read()
formatted_prompt = prompt_template.replace("{degradation_type : severity}", str(pipeline))
model.set_progress_bar_config(disable=None)
qalign = NR_IQA()

image_path = "/home/krishna/workspace/AutoRestore/demo/raw/"
image_paths = glob(os.path.join(image_path,"*.jpg"))
for image_path in image_paths:
    print(f"Processing {image_path}")
    image = Image.open(image_path).convert("RGB")
    inferences_steps = [10,20,30,40,50]
    for steps in inferences_steps:
        inputs = {
            "image": image,
            "prompt": formatted_prompt,
            "generator": torch.manual_seed(0),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": steps,
        }

        with torch.inference_mode():
            output = model(**inputs)
            output_image = output.images[0]
            output_image.save(f"/home/krishna/workspace/AutoRestore/demo/processed/{image_path.split('/')[-1]}-hf-{steps}.jpg")
            qalign_score = qalign.query(str(f"/home/krishna/workspace/AutoRestore/demo/processed/{image_path.split('/')[-1]}-hf-{steps}.jpg"), "qalign")
            print("inference steps:", steps)
            print(f"QAlign score: {qalign_score}")
            print("image saved")
        print("==================================================")
        



