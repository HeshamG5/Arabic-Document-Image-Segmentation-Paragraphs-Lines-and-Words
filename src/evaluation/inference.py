import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Tuple
from torchvision import transforms

from src.utils.image_utils import read_image, resize_image
from src.utils.json_utils import load_json, get_page_polygon
from src.data_preparation.create_masks import create_mask_from_bboxes

def predict_image(
    model: torch.nn.Module,
    image_path: Union[str, Path],
    device: torch.device,
    image_size: Tuple[int, int] = (512, 512),
    threshold: float = 0.5,
    return_original_size: bool = True,
    apply_deskew: bool = True
) -> np.ndarray:
    """
    تطبيق النموذج على صورة واحدة وإرجاع ماسك التقسيم.
    إذا كان return_original_size = True، يتم إعادة تحجيم الماسك إلى أبعاد الصورة الأصلية.
    """
    model.eval()

    # قراءة الصورة
    image = read_image(str(image_path))
    original_h, original_w = image.shape[:2]

    # تحويل إلى tensor وتطبيع
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # تنبؤ
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output) > threshold
        pred = pred.squeeze().cpu().numpy().astype(np.uint8)  # (H, W) مع 0/1

    if return_original_size:
        # إعادة تحجيم الماسك إلى حجم الصورة الأصلي
        pred = cv2.resize(pred, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    return pred


def extract_lines_from_mask(
    mask: np.ndarray,
    original_image: Optional[np.ndarray] = None,
    min_area: int = 100
) -> List[np.ndarray]:
    """
    استخراج السطور من الماسك باستخدام تحليل المكونات المتصلة (connected components).
    تعيد قائمة بصور السطور (أو ماسكاتها).
    """
    from skimage.measure import label, regionprops

    labeled = label(mask, connectivity=2)
    lines = []
    for region in regionprops(labeled):
        if region.area >= min_area:
            # إحداثيات الصندوق المحيط
            minr, minc, maxr, maxc = region.bbox
            if original_image is not None:
                # اقتصاص الصورة الأصلية
                line_crop = original_image[minr:maxr, minc:maxc]
                lines.append(line_crop)
            else:
                # اقتصاص الماسك
                line_crop = mask[minr:maxr, minc:maxc]
                lines.append(line_crop)
    return lines


def predict_folder(
    model: torch.nn.Module,
    input_folder: Union[str, Path],
    output_folder: Union[str, Path],
    device: torch.device,
    image_size: Tuple[int, int] = (512, 512),
    threshold: float = 0.5,
    extract_lines: bool = True,
    save_visualization: bool = True
):
    """
    تطبيق النموذج على جميع الصور في مجلد، وحفظ النتائج.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # المجلدات الفرعية
    masks_dir = output_folder / "masks"
    lines_dir = output_folder / "lines" if extract_lines else None
    vis_dir = output_folder / "visualizations" if save_visualization else None

    masks_dir.mkdir(exist_ok=True)
    if lines_dir:
        lines_dir.mkdir(exist_ok=True)
    if vis_dir:
        vis_dir.mkdir(exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

    for img_path in input_folder.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue

        print(f"Processing {img_path.name}...")

        # تنبؤ الماسك
        mask = predict_image(
            model, img_path, device, image_size, threshold,
            return_original_size=True
        )

        # حفظ الماسك
        mask_filename = img_path.stem + "_mask.png"
        cv2.imwrite(str(masks_dir / mask_filename), mask * 255)

        # استخراج السطور
        if extract_lines and lines_dir:
            original_img = cv2.imread(str(img_path))
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            lines = extract_lines_from_mask(mask, original_image=original_img)
            for i, line_img in enumerate(lines):
                line_filename = f"{img_path.stem}_line_{i:04d}.png"
                cv2.imwrite(str(lines_dir / line_filename),
                            cv2.cvtColor(line_img, cv2.COLOR_RGB2BGR))

        # عرض توضيحي
        if save_visualization and vis_dir:
            # دمج الصورة مع الماسك
            vis = original_img.copy()
            colored_mask = np.zeros_like(original_img)
            colored_mask[:, :, 0] = mask * 255  # أحمر للماسك
            vis = cv2.addWeighted(vis, 0.7, colored_mask, 0.3, 0)
            vis_path = vis_dir / (img_path.stem + "_vis.jpg")
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print("Inference completed.")