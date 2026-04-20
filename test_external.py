import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
import yaml
import matplotlib.pyplot as plt
import sys
from skimage.measure import label, regionprops
import random

sys.path.append(".")
from src.models.unet import UNet

# -------------------- الإعدادات --------------------
MODEL_PATH = "experiments/exp_001/checkpoints/best_model.pth"
CONFIG_PATH = "configs/train_config.yaml"
IMAGE_PATH = r"C:\Users\asus\Desktop\mhurafsh\images\JoM_Kobayat_0445.JPG"
OUTPUT_DIR = "results"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# -------------------- تحميل الإعدادات --------------------
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# -------------------- إعداد الجهاز --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------- إنشاء النموذج --------------------
model = UNet(
    in_channels=config['model']['in_channels'],
    out_channels=config['model']['out_channels'],
    features=config['model']['features']
).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully.")

# -------------------- تحويلات الصورة (بدون Resize ثابت) --------------------
# نقرأ الصورة الأصلية ونحتفظ بأبعادها
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not read image at {IMAGE_PATH}")
original_h, original_w = image.shape[:2]
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# الحجم الذي يتوقعه النموذج (من الإعدادات)
target_h = config['data']['image_height']
target_w = config['data']['image_width']

# دالة لتغيير الحجم مع الحفاظ على نسبة الأبعاد باستخدام padding
def resize_with_padding(img, target_h, target_w, pad_value=0):
    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # إنشاء صورة بالحجم المستهدف مع padding
    padded = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    # حساب موضع الصورة في المنتصف
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return padded, scale, (x_offset, y_offset, new_w, new_h)

# تطبيق padding على الصورة
padded_img, scale, (x_offset, y_offset, new_w, new_h) = resize_with_padding(image_rgb, target_h, target_w, pad_value=0)

# تحويل الصورة المحشوة إلى Tensor وتطبيق Normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(padded_img).unsqueeze(0).to(device)

# -------------------- التنبؤ --------------------
with torch.no_grad():
    output = model(input_tensor)
    pred_mask = torch.sigmoid(output) > 0.5
    pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)  # (target_h, target_w)

# -------------------- إزالة الـ padding من الماسك --------------------
# قص المنطقة الفعلية التي تحتوي على الصورة (بدون padding)
mask_cropped = pred_mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w]

# إعادة تحجيم الماسك إلى حجم الصورة الأصلي
mask_resized = cv2.resize(mask_cropped, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

# -------------------- تحسين الماسك (معالجة مورفولوجية) --------------------
mask_uint8 = (mask_resized * 255).astype(np.uint8)
kernel = np.ones((5,5), np.uint8)
mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
mask_processed = (mask_closed > 0).astype(np.uint8)

# -------------------- استخراج المربعات --------------------
labeled_mask = label(mask_processed, connectivity=2)
all_regions = list(regionprops(labeled_mask))
print(f"Total regions detected: {len(all_regions)}")

# حساب عتبة المساحة ديناميكياً (نسبة من مساحة الصورة الأصلية)
area_threshold_ratio = 0.0001  # 0.01% من مساحة الصورة
dynamic_min_area = max(20, int(original_h * original_w * area_threshold_ratio))
print(f"Dynamic min area: {dynamic_min_area}")

boxes = []
regions = []
for region in all_regions:
    if region.area >= dynamic_min_area:
        minr, minc, maxr, maxc = region.bbox
        boxes.append((minc, minr, maxc, maxr))
        regions.append(region)

# إذا لم يتم العثور على أي مربع، نأخذ أكبر منطقة
if len(boxes) == 0 and len(all_regions) > 0:
    largest = max(all_regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest.bbox
    boxes.append((minc, minr, maxc, maxr))
    regions.append(largest)
    print("No region met threshold. Using largest region.")

print(f"Final lines detected: {len(boxes)}")

# -------------------- حفظ الماسك --------------------
mask_path = Path(OUTPUT_DIR) / "predicted_mask.png"
cv2.imwrite(str(mask_path), mask_processed * 255)

# -------------------- تجميع السطور في فقرات --------------------
def group_lines_into_paragraphs(regions, boxes, vertical_threshold=30):
    if not regions or len(boxes) == 0:
        return []
    line_centers = [(region.bbox[0] + region.bbox[2]) // 2 for region in regions]
    sorted_indices = sorted(range(len(line_centers)), key=lambda i: line_centers[i])
    paragraphs = []
    current_para = [sorted_indices[0]]
    for i in range(1, len(sorted_indices)):
        prev_idx = sorted_indices[i-1]
        curr_idx = sorted_indices[i]
        dist = line_centers[curr_idx] - line_centers[prev_idx]
        if dist <= vertical_threshold:
            current_para.append(curr_idx)
        else:
            paragraphs.append(current_para)
            current_para = [curr_idx]
    if current_para:
        paragraphs.append(current_para)
    return paragraphs

paragraph_groups = group_lines_into_paragraphs(regions, boxes, vertical_threshold=50)

# رسم الفقرات
paragraph_overlay = image_rgb.copy()
para_colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(len(paragraph_groups))]

for para_idx, group in enumerate(paragraph_groups):
    x1 = min(boxes[i][0] for i in group)
    y1 = min(boxes[i][1] for i in group)
    x2 = max(boxes[i][2] for i in group)
    y2 = max(boxes[i][3] for i in group)
    cv2.rectangle(paragraph_overlay, (x1, y1), (x2, y2), para_colors[para_idx], 3)
    cv2.putText(paragraph_overlay, f"P{para_idx+1}", (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, para_colors[para_idx], 2)

para_path = Path(OUTPUT_DIR) / "paragraphs_overlay.png"
cv2.imwrite(str(para_path), cv2.cvtColor(paragraph_overlay, cv2.COLOR_RGB2BGR))
print(f"عدد الفقرات: {len(paragraph_groups)}")

# -------------------- قص وحفظ كل سطر --------------------
lines_dir = Path(OUTPUT_DIR) / "lines"
lines_dir.mkdir(exist_ok=True)

for idx, (x1, y1, x2, y2) in enumerate(boxes):
    line_crop = image_rgb[y1:y2, x1:x2]
    line_crop_bgr = cv2.cvtColor(line_crop, cv2.COLOR_RGB2BGR)
    line_filename = lines_dir / f"line_{idx+1:04d}.png"
    cv2.imwrite(str(line_filename), line_crop_bgr)

print(f"Total lines saved: {len(boxes)}")

# -------------------- رسم المربعات على الصورة --------------------
overlay = image_rgb.copy()
for (x1, y1, x2, y2) in boxes:
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

# -------------------- تلوين السطور --------------------
colored_lines = np.zeros_like(image_rgb, dtype=np.uint8)
for i, region in enumerate(regions):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    mask_single = (labeled_mask == region.label).astype(np.uint8)
    colored_lines[mask_single == 1] = color

alpha = 0.5
image_colored = cv2.addWeighted(image_rgb, 1 - alpha, colored_lines, alpha, 0)

# -------------------- Visualization --------------------
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis('off')
axes[1].imshow(mask_processed, cmap='gray')
axes[1].set_title("Processed Mask")
axes[1].axis('off')
axes[2].imshow(overlay)
axes[2].set_title("Overlay with Bounding Boxes")
axes[2].axis('off')
axes[3].imshow(image_colored)
axes[3].set_title("Colored Lines")
axes[3].axis('off')
plt.tight_layout()

vis_path = Path(OUTPUT_DIR) / "visualization_colored.png"
plt.savefig(vis_path, dpi=150, bbox_inches='tight')
plt.show()