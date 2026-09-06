import torch
import torch.nn.functional as F

def compute_confidence_features(logits):
    """
    logits: tensor به شکل (batch_size, num_classes) — خروجی خام قبل از softmax
    خروجی: دیکشنری شامل top1_score, margin, entropy برای هر نمونه در batch
    """
    probs = F.softmax(logits, dim=-1)  # تبدیل logits به احتمال

    # مرتب‌سازی احتمالات برای پیدا کردن top-1 و top-2
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    top1 = sorted_probs[:, 0]
    top2 = sorted_probs[:, 1]
    margin = top1 - top2

    # آنتروپی: هرچه بیشتر، عدم قطعیت مدل بیشتر
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

    return {
        "top1_score": top1,
        "margin": margin,
        "entropy": entropy,
    }