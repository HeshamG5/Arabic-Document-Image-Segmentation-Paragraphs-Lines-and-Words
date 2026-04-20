import torch

def iou_score(pred, target, smooth=1e-6):
    """Calculate IoU (Intersection over Union)"""
    pred = pred.bool()
    target = target.bool()
    intersection = (pred & target).float().sum((1,2,3))
    union = (pred | target).float().sum((1,2,3))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def dice_score(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum((1,2,3))
    dice = (2. * intersection + smooth) / (pred.sum((1,2,3)) + target.sum((1,2,3)) + smooth)
    return dice.mean().item()