# src/data_preparation/dataset.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
from typing import Tuple, Optional
from torchvision import transforms

class LineSegmentationDataset(Dataset):
    """
    مجموعة بيانات لتقسيم السطور (line segmentation).
    يتوقع وجود الصور والماسكات في مجلدات منفصلة داخل split معين.
    """
    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[transforms.Compose] = None
    ):
        """
        data_dir: المسار إلى مجلد processed (مثل data/processed)
        split: train / val / test
        image_size: أبعاد الصورة بعد إعادة التحجيم (الارتفاع، العرض)
        transform: تحويلات إضافية للصورة (اختياري)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.transform = transform

        # مسارات الصور والماسكات
        self.images_dir = self.data_dir / split / "images"
        self.masks_dir = self.data_dir / split / "masks"

        # قائمة بجميع أسماء الملفات (بدون امتداد)
        self.image_files = [f.stem for f in self.images_dir.glob("*.png")]
        self.image_files.sort()

        if not self.image_files:
            raise RuntimeError(f"No images found in {self.images_dir}")

        # تحويل أساسي لتحويل الصورة إلى توتر وتطبيع
        self.default_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # قراءة الصورة
        img_path = self.images_dir / f"{self.image_files[idx]}.png"
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # قراءة الماسك
        mask_path = self.masks_dir / f"{self.image_files[idx]}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        # إعادة تحجيم الصورة والماسك
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]),
                           interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]),
                          interpolation=cv2.INTER_NEAREST)

        # تحويل الماسك إلى ثنائي (0/1)
        mask = (mask > 127).astype(np.float32)

        # تطبيق التحويلات على الصورة
        if self.transform:
            image = self.transform(image)
        else:
            # تحويل الصورة إلى توتر مع تطبيع
            image = self.default_transform(image)

        # تحويل الماسك إلى توتر
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        return image, mask