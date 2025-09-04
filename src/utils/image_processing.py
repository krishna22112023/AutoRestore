import logging
import base64
import mimetypes
from PIL import Image

logger = logging.getLogger(__name__)

def resize_image(img_response,height,width):
    try:
        import torch
        import torch.nn.functional as F
        import torchvision.transforms as T
        from io import BytesIO
        import os

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert the response bytes to a PIL image
        response_pil = Image.open(BytesIO(img_response.content)).convert("RGB")
        tensor_img = T.ToTensor()(response_pil).unsqueeze(0).to(device)

        # Resize tensor to original (height, width)
        resized_tensor = F.interpolate(tensor_img, size=(height, width), mode="bilinear", align_corners=False)

        # Convert back to PIL and save
        resized_pil = T.ToPILImage()(resized_tensor.squeeze(0).cpu())
        return resized_pil
    except Exception as e:
        logger.error(f"Failed to resize with torch: {e}. Saving original image.")
        return img_response.content

def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("Unsupported or unrecognized image format")
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"