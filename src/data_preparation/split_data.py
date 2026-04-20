import os
import random
from pathlib import Path

def create_splits(
    images_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    إنشاء ملفات تقسيم البيانات (train.txt, val.txt, test.txt)
    بناءً على قائمة الصور الموجودة في images_dir.
    """
    random.seed(seed)
    # الحصول على جميع الصور (بدون امتداد)
    image_files = [
        f.stem for f in images_dir.glob('*')
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ]
    random.shuffle(image_files)

    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'train.txt', 'w') as f:
        f.write('\n'.join(train_files))
    with open(output_dir / 'val.txt', 'w') as f:
        f.write('\n'.join(val_files))
    with open(output_dir / 'test.txt', 'w') as f:
        f.write('\n'.join(test_files))

    print(f"Splits created: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

if __name__ == "__main__":
    # مثال للاستخدام
    create_splits(
        images_dir=Path("data/raw/images"),
        output_dir=Path("data/splits")
    )