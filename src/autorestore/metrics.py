'''
- Retoration score : cos_sim * gamma + qalign_score * (1-gamma)
- Full reference IQA comparison : PSNR, SSIM, LPIPS
- No reference IQA comparison : MANIQA, CLIP-QA, MUSIQ
'''

from pathlib import Path
from src.autorestore.IQA import ViT, QAlign 

def restoration_score(raw_image: Path, proc_image: Path, vit: ViT, qalign: QAlign, _lambda: float) -> float:
    #calculate scores
    cos_sim = vit.query(str(raw_image), str(proc_image))
    qalign_score = qalign.query(str(proc_image))
    #min max normalize score
    cos_sim_normalized = (cos_sim + 1) / 2
    qalign_score_normalized = (qalign_score - 1) / 4
    score = _lambda * cos_sim_normalized + (1-_lambda) * qalign_score_normalized
    return score