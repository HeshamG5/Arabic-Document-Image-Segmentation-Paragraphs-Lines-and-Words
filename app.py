import streamlit as st
import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
import yaml
from skimage.measure import label, regionprops
import random
from PIL import Image
import zipfile
import io
import shutil
from datetime import datetime
import re
import sys
import traceback

# -------------------- Import your model architecture --------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from src.models.unet import UNet

# -------------------- Configuration --------------------
MODEL_PATH = "C:/Users/asus/Desktop/line_seg/experiments/exp_001/checkpoints/best_model.pth"
CONFIG_PATH = "C:/Users/asus/Desktop/line_seg/configs/train_config.yaml"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

CROP_DIR = Path("cropped_paragraphs")
CROP_DIR.mkdir(exist_ok=True)

LINE_DETECT_DIR = Path("line_detected_paragraphs")
LINE_DETECT_DIR.mkdir(exist_ok=True)

CROPPED_LINES_DIR = Path("cropped_lines")
CROPPED_LINES_DIR.mkdir(exist_ok=True)

# -------------------- Streamlit page setup --------------------
st.set_page_config(page_title="Segmentation Interface", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    .img-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(90deg, #e67e22, #f39c12);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(230, 126, 34, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🖼️ Document Line & Paragraph & Word Segmentation")
st.write("Upload an image and see the segmentation results")

# -------------------- Load UNet model (cached) --------------------
@st.cache_resource
def load_unet_model():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        features=config['model']['features']
    ).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device, config

# -------------------- Load YOLOv5 model using torch.hub with GPU --------------------
@st.cache_resource
def load_line_model():
    model_path = Path("line_model_best.pt")
    if not model_path.exists():
        model_path = Path(__file__).parent / "line_model_best.pt"
    if not model_path.exists():
        st.error("❌ خطأ: لم يتم العثور على ملف نموذج line_model_best.pt")
        return None, None
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path), force_reload=False)
        model.to(device)
        if hasattr(model, 'model'):
            model.model.to(device)
        model.conf = 0.25
        model.iou = 0.45
        st.info(f"✅ نموذج الكشف عن الأسطر تم تحميله على: {device}")
        return model, device
    except Exception as e:
        st.error(f"خطأ في تحميل نموذج YOLO: {e}")
        return None, None

# -------------------- Helper functions for paragraph segmentation (unchanged) --------------------
def resize_with_padding(img, target_h, target_w, pad_value=0):
    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return padded, scale, (x_offset, y_offset, new_w, new_h)

def predict_mask(model, device, image, config):
    target_h = config['data']['image_height']
    target_w = config['data']['image_width']
    padded_img, scale, (x_offset, y_offset, new_w, new_h) = resize_with_padding(image, target_h, target_w)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(padded_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output) > 0.5
        pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
    mask_cropped = pred_mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
    original_h, original_w = image.shape[:2]
    mask_resized = cv2.resize(mask_cropped, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    return mask_resized

def postprocess_mask(mask, kernel_size=5, close_iter=1):
    mask_uint8 = (mask * 255).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    return (mask_closed > 0).astype(np.uint8)

def extract_boxes(mask, min_area_ratio=0.0001):
    labeled = label(mask, connectivity=2)
    regions = list(regionprops(labeled))
    if not regions:
        return [], [], labeled
    h, w = mask.shape
    min_area = max(20, int(h * w * min_area_ratio))
    boxes = []
    valid_regions = []
    for region in regions:
        if region.area >= min_area:
            minr, minc, maxr, maxc = region.bbox
            boxes.append((minc, minr, maxc, maxr))
            valid_regions.append(region)
    if not boxes:
        largest = max(regions, key=lambda r: r.area)
        minr, minc, maxr, maxc = largest.bbox
        boxes.append((minc, minr, maxc, maxr))
        valid_regions.append(largest)
    return boxes, valid_regions, labeled

def group_lines_into_paragraphs(regions, boxes, vertical_threshold=30):
    if not regions or not boxes:
        return []
    centers = [(region.bbox[0] + region.bbox[2]) // 2 for region in regions]
    sorted_indices = sorted(range(len(centers)), key=lambda i: centers[i])
    paragraphs = []
    current = [sorted_indices[0]]
    for i in range(1, len(sorted_indices)):
        prev_idx = sorted_indices[i-1]
        curr_idx = sorted_indices[i]
        if centers[curr_idx] - centers[prev_idx] <= vertical_threshold:
            current.append(curr_idx)
        else:
            paragraphs.append(current)
            current = [curr_idx]
    if current:
        paragraphs.append(current)
    return paragraphs

def draw_boxes(image, boxes, color=(255, 0, 0), thickness=2):
    img_copy = image.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
    return img_copy

def color_lines(image, labeled_mask, regions):
    colored = np.zeros_like(image, dtype=np.uint8)
    for region in regions:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        mask_single = (labeled_mask == region.label).astype(np.uint8)
        colored[mask_single == 1] = color
    return cv2.addWeighted(image, 0.6, colored, 0.4, 0)

def draw_paragraphs(image, paragraphs, boxes):
    img_copy = image.copy()
    colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) 
              for _ in range(len(paragraphs))]
    for para_idx, group in enumerate(paragraphs):
        x1 = min(boxes[i][0] for i in group)
        y1 = min(boxes[i][1] for i in group)
        x2 = max(boxes[i][2] for i in group)
        y2 = max(boxes[i][3] for i in group)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), colors[para_idx], 3)
        cv2.putText(img_copy, f"P{para_idx+1}", (x1+5, y1+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[para_idx], 2)
    return img_copy

def crop_and_save_paragraphs(image, paragraphs, boxes, save_dir, clear_first=True):
    if clear_first and save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cropped = []
    for para_idx, group in enumerate(paragraphs):
        x1 = min(boxes[i][0] for i in group)
        y1 = min(boxes[i][1] for i in group)
        x2 = max(boxes[i][2] for i in group)
        y2 = max(boxes[i][3] for i in group)
        crop = image[y1:y2, x1:x2]
        cropped.append((para_idx, crop))
        save_path = save_dir / f"paragraph_{para_idx+1}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    return cropped

def create_zip_from_images(images, zip_name="paragraphs.zip"):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for idx, img in images:
            img_pil = Image.fromarray(img)
            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format="PNG")
            zip_file.writestr(f"paragraph_{idx+1}.png", img_bytes.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

# -------------------- New function: extract and save lines from a paragraph image --------------------
def extract_and_save_lines(paragraph_img_bgr, boxes, output_dir, prefix):
    cropped_lines = []
    for i, (x1, y1, x2, y2, score) in enumerate(boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        line_crop = paragraph_img_bgr[y1:y2, x1:x2]
        line_rgb = cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB)
        out_path = output_dir / f"{prefix}_line_{i+1}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(line_rgb, cv2.COLOR_RGB2BGR))
        cropped_lines.append((i, line_rgb))
    return cropped_lines

# -------------------- Line Detection functions (original) --------------------
def preprocess_versions(image_bgr):
    versions = []
    rgb_original = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    versions.append(('original', rgb_original))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    versions.append(('gray', gray_rgb))
    eq = cv2.equalizeHist(gray)
    eq_rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
    versions.append(('equalize', eq_rgb))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    clahe_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    versions.append(('clahe', clahe_rgb))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_rgb = cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB)
    versions.append(('otsu', otsu_rgb))
    return versions

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    indices = np.argsort(scores)[::-1]
    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1:
            break
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        x1 = np.maximum(current_box[0], other_boxes[:, 0])
        y1 = np.maximum(current_box[1], other_boxes[:, 1])
        x2 = np.minimum(current_box[2], other_boxes[:, 2])
        y2 = np.minimum(current_box[3], other_boxes[:, 3])
        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)
        inter = w * h
        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_others = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        union = area_current + area_others - inter
        iou = inter / (union + 1e-6)
        mask = iou < iou_threshold
        indices = indices[1:][mask]
    return keep

def detect_lines_on_image(image_path, model, device):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return [], None

    versions = preprocess_versions(img_bgr)
    conf_thresholds = [0.5, 0.25, 0.1]
    sizes = [640, 1280]

    all_boxes = []
    all_scores = []

    for version_name, img_rgb in versions:
        for conf in conf_thresholds:
            for size in sizes:
                model.conf = conf
                results = model(img_rgb, size=size, augment=True)
                boxes = results.xyxy[0].cpu().numpy()
                if len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2, score, cls = box
                        all_boxes.append([x1, y1, x2, y2])
                        all_scores.append(score)

    if len(all_boxes) == 0:
        return [], None

    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    keep = non_max_suppression(all_boxes, all_scores, iou_threshold=0.5)
    final_boxes = all_boxes[keep]
    final_scores = all_scores[keep]

    img_draw = img_bgr.copy()
    for (x1, y1, x2, y2), score in zip(final_boxes, final_scores):
        cv2.rectangle(img_draw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{score:.2f}"
        cv2.putText(img_draw, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    drawn_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    boxes_list = [(box[0], box[1], box[2], box[3], score) for box, score in zip(final_boxes, final_scores)]
    return boxes_list, drawn_rgb

def process_all_paragraphs(crop_dir, model, device, output_dir, clear_first=True):
    if clear_first and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines_output_dir = CROPPED_LINES_DIR
    if clear_first and lines_output_dir.exists():
        shutil.rmtree(lines_output_dir)
    lines_output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(crop_dir.glob("*.png")) + list(crop_dir.glob("*.jpg")) + list(crop_dir.glob("*.jpeg"))
    if not image_files:
        return [], []

    progress_bar = st.progress(0, text="Detecting lines...")
    status_text = st.empty()
    processed = []
    all_cropped_lines = []

    for idx, img_path in enumerate(image_files):
        status_text.text(f"Processing {img_path.name} ({idx+1}/{len(image_files)})")
        boxes, drawn = detect_lines_on_image(img_path, model, device)
        if drawn is not None:
            out_path = output_dir / f"{img_path.stem}_detected.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
            processed.append((img_path.stem, drawn))

            orig_img = cv2.imread(str(img_path))
            if orig_img is not None and boxes:
                for line_idx, (x1, y1, x2, y2, score) in enumerate(boxes):
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    h, w = orig_img.shape[:2]
                    x1, x2 = max(0, x1), min(w, x2)
                    y1, y2 = max(0, y1), min(h, y2)
                    line_crop = orig_img[y1:y2, x1:x2]
                    if line_crop.size == 0:
                        continue
                    line_rgb = cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB)
                    filename = f"{img_path.stem}_line_{line_idx+1}.png"
                    save_path = lines_output_dir / filename
                    cv2.imwrite(str(save_path), cv2.cvtColor(line_rgb, cv2.COLOR_RGB2BGR))
                    all_cropped_lines.append((line_rgb, filename))
        progress_bar.progress((idx + 1) / len(image_files))

    status_text.empty()
    progress_bar.empty()
    return processed, all_cropped_lines

# -------------------- Main app with session state --------------------
def main():
    # ---- Session state initializations ----
    if "segmentation_done" not in st.session_state:
        st.session_state.segmentation_done = False
    if "line_detection_done" not in st.session_state:
        st.session_state.line_detection_done = False
    if "line_zip_buffer" not in st.session_state:
        st.session_state.line_zip_buffer = None
    if "cropped_lines" not in st.session_state:
        st.session_state.cropped_lines = None
    if "original_image" not in st.session_state:
        st.session_state.original_image = None   # <-- NEW: store original image

    unet_model, device, config = load_unet_model()
    st.success("✅ UNet model loaded successfully")

    line_model, line_device = load_line_model()
    if line_model is None:
        st.warning("⚠️ نموذج الكشف عن الأسطر غير متوفر. تأكد من وجود ملف line_model_best.pt")
    else:
        st.info(f"نموذج YOLOv5 يعمل على: {line_device}")

    with st.sidebar:
        st.header("⚙️ Segmentation Parameters")
        threshold = st.slider("Mask threshold (sigmoid >)", 0.0, 1.0, 0.5, 0.05)
        kernel_size = st.slider("Morphology kernel size", 1, 15, 5, step=2)
        close_iter = st.slider("Closing iterations", 1, 5, 1)
        min_area_ratio = st.number_input("Min area ratio", 0.0, 0.1, 0.0001, format="%.6f")
        vertical_thresh = st.slider("Vertical threshold for paragraphs", 10, 100, 30)

        st.divider()
        st.header("📁 Output Management")
        if st.button("🗑️ Delete all cropped paragraphs now"):
            if CROP_DIR.exists():
                shutil.rmtree(CROP_DIR)
                CROP_DIR.mkdir(exist_ok=True)
                st.success("All cropped paragraphs deleted.")
                st.session_state.segmentation_done = False
                st.session_state.line_detection_done = False
        if st.button("🗑️ Delete all line detection results (detected images & cropped lines)"):
            if LINE_DETECT_DIR.exists():
                shutil.rmtree(LINE_DETECT_DIR)
                LINE_DETECT_DIR.mkdir(exist_ok=True)
            if CROPPED_LINES_DIR.exists():
                shutil.rmtree(CROPPED_LINES_DIR)
                CROPPED_LINES_DIR.mkdir(exist_ok=True)
            st.success("All line detection results deleted.")
            st.session_state.line_detection_done = False
            st.session_state.cropped_lines = None

    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        # Save original image in session state for later use in word segmentation page
        st.session_state.original_image = img_np

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.info("🖼️ Original Image")
            st.image(img_np, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🚀 Run Segmentation"):
            with st.spinner("Running segmentation..."):
                mask = predict_mask(unet_model, device, img_np, config)
                if threshold != 0.5:
                    mask = (mask > threshold).astype(np.uint8)
                mask_proc = postprocess_mask(mask, kernel_size=kernel_size, close_iter=close_iter)
                boxes, regions, labeled = extract_boxes(mask_proc, min_area_ratio=min_area_ratio)
                paragraphs = group_lines_into_paragraphs(regions, boxes, vertical_threshold=vertical_thresh)

                overlay_boxes = draw_boxes(img_np, boxes)
                colored_lines_img = color_lines(img_np, labeled, regions)
                paragraph_img = draw_paragraphs(img_np, paragraphs, boxes)
                mask_display = (mask_proc * 255).astype(np.uint8)

                with col2:
                    st.markdown('<div class="img-container">', unsafe_allow_html=True)
                    st.success("🧪 Segmentation Results")
                    st.image(mask_display, caption="Predicted Mask", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.subheader("📋 Detailed Results")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.image(overlay_boxes, caption="Bounding Boxes", use_container_width=True)
                    st.image(colored_lines_img, caption="Colored Lines", use_container_width=True)
                with res_col2:
                    st.image(paragraph_img, caption="Paragraph Groups", use_container_width=True)
                    st.image(mask_display, caption="Predicted Mask", use_container_width=True)

                st.divider()
                st.subheader("📊 Statistics")
                st.write(f"**Number of lines detected:** {len(boxes)}")
                st.write(f"**Number of paragraphs:** {len(paragraphs)}")
                if paragraphs:
                    st.write("**Paragraph groups:**")
                    for i, group in enumerate(paragraphs):
                        st.write(f"  Paragraph {i+1}: lines {[j+1 for j in group]}")

                if paragraphs:
                    cropped_paras = crop_and_save_paragraphs(img_np, paragraphs, boxes, CROP_DIR, clear_first=True)
                    st.session_state.cropped_paragraphs_count = len(cropped_paras)
                    st.session_state.segmentation_done = True
                    st.success(f"✅ Saved {len(cropped_paras)} paragraph images to `{CROP_DIR}` (previous ones replaced).")

                    zip_buffer = create_zip_from_images(cropped_paras, "paragraphs.zip")
                    st.download_button(
                        label="📦 Download these paragraphs as ZIP",
                        data=zip_buffer,
                        file_name=f"paragraphs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )

                st.divider()
                st.subheader("💾 Download Mask")
                mask_pil = Image.fromarray(mask_display)
                st.download_button("Download Mask (PNG)", mask_pil.tobytes(), "mask.png", "image/png")

        if st.session_state.segmentation_done and line_model is not None:
            st.divider()
            if st.button("🔍 Run Line Segmentation on Cropped Paragraphs", key="line_detection_btn"):
                with st.spinner("Detecting lines in all paragraph images and cropping lines..."):
                    try:
                        processed, cropped_lines = process_all_paragraphs(CROP_DIR, line_model, line_device, LINE_DETECT_DIR, clear_first=True)
                        if processed:
                            st.session_state.line_detection_done = True
                            st.session_state.cropped_lines = cropped_lines
                            zip_images = [(i, img) for i, (_, img) in enumerate(processed)]
                            st.session_state.line_zip_buffer = create_zip_from_images(zip_images, "line_detected_paragraphs.zip")
                        else:
                            st.warning("No images found in cropped_paragraphs or no lines detected.")
                    except Exception as e:
                        st.error(f"خطأ أثناء معالجة الأسطر: {e}")
                        st.error(traceback.format_exc())

            if st.session_state.line_detection_done and st.session_state.line_zip_buffer is not None:
                st.success(f"✅ Saved {len(list(LINE_DETECT_DIR.glob('*_detected.png')))} line-detected images to `{LINE_DETECT_DIR}`.")
                st.success(f"✅ Saved cropped lines to `{CROPPED_LINES_DIR}`.")

                detected_images = list(LINE_DETECT_DIR.glob("*_detected.png"))
                if detected_images:
                    st.subheader("🔍 Sample Line Detection Results (with bounding boxes)")
                    sample_cols = st.columns(min(3, len(detected_images)))
                    for idx, img_path in enumerate(detected_images[:3]):
                        with sample_cols[idx]:
                            st.image(str(img_path), caption=img_path.name, use_container_width=True)

                st.download_button(
                    label="📦 Download all line-detected images (with boxes) as ZIP",
                    data=st.session_state.line_zip_buffer,
                    file_name=f"line_detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )

                # Show cropped lines and provide navigation button instead of download
                if st.session_state.cropped_lines and len(st.session_state.cropped_lines) > 0:
                    st.subheader("✂️ Cropped Lines (individual lines)")
                    line_display_cols = st.columns(4)
                    for idx, (line_img, filename) in enumerate(st.session_state.cropped_lines):
                        col = idx % 4
                        with line_display_cols[col]:
                            st.image(line_img, caption=filename, use_container_width=True)

                    # ------------------ NEW BUTTON ------------------
                    st.markdown("---")
                    st.markdown("### 📄 Go to Word Segmentation")
                    st.write("After line detection, you can now segment the **original image** into words.")
                    if st.button("➡️ Go to Word Segmentation Page", use_container_width=True):
                        # This will navigate to the page /word_segmentation
                        st.switch_page("pages/word_segmentation.py")
                    # ------------------------------------------------
                else:
                    st.info("No cropped lines available.")

if __name__ == "__main__":
    main()