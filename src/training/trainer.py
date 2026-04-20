import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import platform

from src.models.unet import UNet
from src.models.loss import DiceLoss, CombinedLoss
from src.models.metrics import iou_score, dice_score

# تفعيل الـ cuDNN auto-tuner لتسريع الأداء على GPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

class Trainer:
    def __init__(
        self,
        config: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: nn.Module,
        device: torch.device,
        experiment_path: Path
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.device = device
        self.experiment_path = experiment_path
        self.checkpoint_dir = experiment_path / config.get('checkpoint_dir', 'checkpoints')
        self.log_file = experiment_path / 'training_log.txt'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # إعداد دالة الخسارة
        loss_name = config.get('loss', 'dice_loss')
        if loss_name == 'dice_loss':
            self.criterion = DiceLoss()
        elif loss_name == 'combined_loss':
            self.criterion = CombinedLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # إعداد المحسن
        optimizer_name = config.get('optimizer', 'adam').lower()
        lr = config.get('learning_rate', 0.001)
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # جدولة معدل التعلم
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Mixed Precision Scaler (بالصيغة الجديدة لـ PyTorch)
        if device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        self.epochs = config.get('epochs', 100)
        self.start_epoch = 0
        self.best_iou = 0.0

    def log(self, message: str):
        """كتابة رسالة إلى ملف السجل وطباعتها"""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).float()

            self.optimizer.zero_grad()

            if self.scaler:
                # Mixed precision with new syntax (no warnings)
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

            preds = torch.sigmoid(outputs) > 0.5
            iou = iou_score(preds, masks)
            dice = dice_score(preds, masks)

            total_loss += loss.item()
            total_iou += iou
            total_dice += dice

            pbar.set_postfix({'loss': loss.item(), 'iou': iou, 'dice': dice})

        avg_loss = total_loss / len(self.train_loader)
        avg_iou = total_iou / len(self.train_loader)
        avg_dice = total_dice / len(self.train_loader)

        return {'loss': avg_loss, 'iou': avg_iou, 'dice': avg_dice}

    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]")
        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True).float()

                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                preds = torch.sigmoid(outputs) > 0.5
                iou = iou_score(preds, masks)
                dice = dice_score(preds, masks)

                total_loss += loss.item()
                total_iou += iou
                total_dice += dice

        avg_loss = total_loss / len(self.val_loader)
        avg_iou = total_iou / len(self.val_loader)
        avg_dice = total_dice / len(self.val_loader)

        return {'loss': avg_loss, 'iou': avg_iou, 'dice': avg_dice}

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_iou': self.best_iou,
            'config': self.config
        }
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
        else:
            path = self.checkpoint_dir / f'epoch_{epoch+1:03d}.pth'
        torch.save(checkpoint, path)
        self.log(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.best_iou = checkpoint['best_iou']
        self.log(f"Loaded checkpoint from {path}, epoch {self.start_epoch}")

    def fit(self):
        self.log("Starting training...")
        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch)

            # تسجيل المقاييس
            self.log(f"Epoch {epoch+1}: Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train IoU: {train_metrics['iou']:.4f}, Train Dice: {train_metrics['dice']:.4f}")
            self.log(f"           Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val IoU: {val_metrics['iou']:.4f}, Val Dice: {val_metrics['dice']:.4f}")

            self.scheduler.step(val_metrics['loss'])

            if val_metrics['iou'] > self.best_iou:
                self.best_iou = val_metrics['iou']
                self.save_checkpoint(epoch, is_best=True)

            if (epoch + 1) % self.config.get('save_frequency', 5) == 0:
                self.save_checkpoint(epoch)

        self.log("Training completed.")


if __name__ == "__main__":
    import argparse
    import yaml
    from torch.utils.data import DataLoader
    from pathlib import Path
    from src.data_preparation.dataset import LineSegmentationDataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("Config loaded. Sections:", list(config.keys()))

    # إعداد مسار التجربة
    experiment_path = Path(config['experiment']['path']) / config['experiment']['name']
    experiment_path.mkdir(parents=True, exist_ok=True)
    print(f"Experiment path: {experiment_path}")

    # تحديد الجهاز
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

    # تحميل البيانات
    data_dir = Path(config['data']['processed_dir'])
    image_size = (config['data']['image_height'], config['data']['image_width'])

    train_dataset = LineSegmentationDataset(
        data_dir=data_dir,
        split='train',
        image_size=image_size,
        transform=None
    )
    val_dataset = LineSegmentationDataset(
        data_dir=data_dir,
        split='val',
        image_size=image_size,
        transform=None
    )

    # إعداد DataLoader
    num_workers = config['training'].get('num_workers', 0)
    if platform.system() == 'Windows' and num_workers > 0:
        print("Warning: On Windows, setting num_workers=0 for stability.")
        num_workers = 0

    pin_memory = (device.type == 'cuda') and (num_workers > 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # إنشاء النموذج
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        features=config['model']['features']
    )

    # تعطيل torch.compile على Windows (حيث لا يعمل)
    # if hasattr(torch, 'compile') and platform.system() != 'Windows':
    #     model = torch.compile(model)
    #     print("Using torch.compile for faster training")

    print("Model created")

    # إنشاء المدرب والبدء
    trainer = Trainer(
        config=config['training'],
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=device,
        experiment_path=experiment_path
    )
    trainer.load_checkpoint('experiments/exp_001/checkpoints/epoch_005.pth')
    trainer.fit()