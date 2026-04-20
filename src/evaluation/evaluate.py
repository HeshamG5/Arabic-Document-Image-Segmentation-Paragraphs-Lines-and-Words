import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional

from src.models.metrics import iou_score, dice_score
from src.utils.metrics_utils import iou_score as iou_np, dice_score as dice_np

def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    تقييم النموذج على مجموعة الاختبار.
    يعيد متوسطات IoU و Dice.
    """
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    num_batches = len(test_loader)

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs) > threshold

            # حساب المقاييس باستخدام دوال من models.metrics (تعمل على tensors)
            iou = iou_score(preds, masks)
            dice = dice_score(preds, masks)

            total_iou += iou.item()
            total_dice += dice.item()

    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches

    return {"IoU": avg_iou, "Dice": avg_dice}


def evaluate_model_with_details(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, List[float]]:
    """
    تقييم النموذج وإرجاع قائمة المقاييس لكل صورة (لمزيد من التحليل الإحصائي).
    """
    model.eval()
    iou_list = []
    dice_list = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating per image"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs) > threshold

            # نقل إلى CPU ثم numpy
            preds_np = preds.cpu().numpy().astype(np.uint8)
            masks_np = masks.cpu().numpy().astype(np.uint8)

            for i in range(preds_np.shape[0]):
                iou = iou_np(preds_np[i, 0], masks_np[i, 0])
                dice = dice_np(preds_np[i, 0], masks_np[i, 0])
                iou_list.append(iou)
                dice_list.append(dice)

    return {"IoU": iou_list, "Dice": dice_list}