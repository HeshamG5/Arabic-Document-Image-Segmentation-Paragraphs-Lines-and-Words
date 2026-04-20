import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.utils.json_utils import load_json, get_text_lines_bboxes, get_page_polygon

def create_mask_from_bboxes(
    bboxes: list,
    img_shape: tuple,
    page_polygon: list = None
) -> np.ndarray:
    """
    إنشاء ماسك ثنائي من مستطيلات السطور.
    إذا وُجد مضلع الصفحة، يتم قص الماسك به.
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=cv2.FILLED)

    if page_polygon:
        poly_np = np.array(page_polygon, dtype=np.int32)
        poly_mask = np.zeros_like(mask)
        cv2.fillPoly(poly_mask, [poly_np], 1)
        mask = mask * poly_mask

    return mask

def process_dataset(
    raw_dir: Path,
    processed_dir: Path,
    split: str,
    split_files: list
):
    """
    معالجة مجموعة بيانات معينة (train/val/test) وتحويلها إلى صور وماسكات.
    """
    images_out = processed_dir / split / "images"
    masks_out = processed_dir / split / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    for name in tqdm(split_files, desc=f"Processing {split}"):
        # البحث عن الصورة (بأي امتداد)
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            p = raw_dir / "images" / f"{name}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            print(f"Warning: Image not found for {name}, skipping.")
            continue

        json_path = raw_dir / "annotations" / f"{name}.json"
        if not json_path.exists():
            print(f"Warning: JSON not found for {name}, skipping.")
            continue

        # قراءة الصورة
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # تحميل التعليقات
        try:
            ann = load_json(str(json_path))
        except Exception as e:
            print(f"Error loading JSON for {name}: {e}")
            continue

        bboxes = get_text_lines_bboxes(ann)
        if not bboxes:
            print(f"Warning: No text lines found in {name}, skipping.")
            continue

        page_poly = get_page_polygon(ann)

        # إنشاء الماسك
        mask = create_mask_from_bboxes(bboxes, img.shape, page_poly)

        # حفظ الصورة والماسك
        out_img_path = images_out / f"{name}.png"
        out_mask_path = masks_out / f"{name}.png"
        cv2.imwrite(str(out_img_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_mask_path), mask * 255)

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    splits_dir = Path("data/splits")

    # قراءة ملفات التقسيم
    train_files = [line.strip() for line in open(splits_dir / "train.txt")]
    val_files = [line.strip() for line in open(splits_dir / "val.txt")]
    test_files = [line.strip() for line in open(splits_dir / "test.txt")]

    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")

    process_dataset(raw_dir, processed_dir, "train", train_files)
    process_dataset(raw_dir, processed_dir, "val", val_files)
    process_dataset(raw_dir, processed_dir, "test", test_files)