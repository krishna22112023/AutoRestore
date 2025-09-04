'''
- Retoration score : cos_sim * gamma + qalign_score * (1-gamma)
- Full reference IQA comparison : PSNR, SSIM, LPIPS
- No reference IQA comparison : MANIQA, CLIP-QA, MUSIQ
'''

from pathlib import Path
from src.autorestore.IQA import ViT, QAlign 

def restoration_score(img_path: Path, current_img_path: Path, vit: ViT, qalign: QAlign, gamma: float = 0.5) -> float:
    #calculate scores
    cos_sim = vit.query(str(img_path), str(current_img_path))
    qalign_score = qalign.query(str(current_img_path))
    #min max normalize score
    cos_sim_normalized = (cos_sim + 1) / 2
    qalign_score_normalized = (qalign_score - 1) / 4
    score = gamma * cos_sim_normalized + (1-gamma) * qalign_score_normalized
    return score