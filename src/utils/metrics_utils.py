import numpy as np

def iou_score(pred: np.ndarray, target: np.ndarray, smooth=1e-6) -> float:
    """
    حساب IoU (Intersection over Union) لماسكين ثنائيين.
    pred, target: مصفوفات ثنائية (0/1) بنفس الأبعاد.
    """
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return (intersection + smooth) / (union + smooth)

def dice_score(pred: np.ndarray, target: np.ndarray, smooth=1e-6) -> float:
    """حساب Dice coefficient."""
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def accuracy_score(pred: np.ndarray, target: np.ndarray) -> float:
    """دقة التصنيف على مستوى البكسل."""
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    correct = (pred == target).sum()
    total = pred.size
    return correct / total