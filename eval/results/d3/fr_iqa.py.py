from src.autorestore.IQA import NR_IQA,psnr,ssim
from pathlib import Path
from config import settings
from src.utils.image_processing import resize_image
import json
supported_exts = ["png"]
folder = ["rain_noise_low_resolution"] 
lambda_ = ["lambda_0","lambda_0_3","lambda_0_5","lambda_0_7"]
nr_iqa = NR_IQA()
for fol in folder:
    for lam in lambda_:
        patterns = [f"*.{e}" for e in supported_exts]
        processed_path = Path(settings.BASE / f"eval/data/d3/{fol}/{lam}/processed")
        verify_IQA_path = Path(settings.BASE / f"eval/data/d3/{fol}/{lam}/artefacts/verify_IQA.json")
        print("processed path", processed_path)
        print("verify_IQA_path", verify_IQA_path)
        with open(verify_IQA_path, "r") as f:
            verify_IQA = json.load(f)
        print("vrify_IQA path loaded")
        imgs = [p for pat in patterns for p in processed_path.glob(pat)]
        for img in imgs:
            print("processing", img.name)
            proc_img = processed_path / img.name
            gnd_img = Path(settings.BASE / f"eval/gnd_truth") / img.name
            psnr_score = psnr(str(gnd_img), str(proc_img))
            ssim_score = ssim(str(gnd_img), str(proc_img))
            lpips_score = nr_iqa.query(str(gnd_img), "lpips",str(proc_img))
            print(f"{img.name} - PSNR: {psnr_score}, SSIM: {ssim_score}, LPIPS: {lpips_score}")
            verify_IQA[img.name]["FR_IQA"] = {
                "psnr": psnr_score,
                "ssim": ssim_score,
                "lpips": lpips_score
            }
        with open(verify_IQA_path, "w") as f:
            json.dump(verify_IQA, f,indent=4)
        print("verify_IQA written")

