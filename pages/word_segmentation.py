import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
from datetime import datetime
import traceback

try:
    from paddleocr import PaddleOCR
except ImportError:
    st.error("PaddleOCR not installed. Please install it: pip install paddlepaddle paddleocr")
    st.stop()

# Set page config (must be the first Streamlit command)
st.set_page_config(page_title="Word Segmentation", layout="wide")

st.title("🔤 Word Segmentation on Original Image")
st.write("Detect individual words and extract cropped images of each word.")

# Initialize OCR engine only once (cached) – force CPU
@st.cache_resource
def load_word():
    # Use English language; change to 'ar' for Arabic if needed
    # force CPU by setting use_gpu=False
    ocr = PaddleOCR(use_angle_cls=True, lang='ar', use_gpu=False)
    return ocr

def draw_word_boxes(image, boxes):
    """Draw bounding boxes around words and return the image with boxes."""
    img_copy = image.copy()
    for box in boxes:
        # box is a list of four points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_copy, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return img_copy

def crop_word_images(image, boxes):
    """Crop each word bounding box and return a list of (index, cropped_image)."""
    cropped = []
    for idx, box in enumerate(boxes):
        # Get bounding rectangle from the 4 points
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        x1, y1 = int(min(x_coords)), int(min(y_coords))
        x2, y2 = int(max(x_coords)), int(max(y_coords))
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        word_crop = image[y1:y2, x1:x2]
        cropped.append((idx, word_crop))
    return cropped

def create_zip_from_word_images(images, zip_name="words.zip"):
    """Create a ZIP file from a list of (index, image_array) tuples."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for idx, img in images:
            img_pil = Image.fromarray(img)
            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format="PNG")
            zip_file.writestr(f"word_{idx+1}.png", img_bytes.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

def main():
    # Check if original image is stored in session state
    if "original_image" not in st.session_state or st.session_state.original_image is None:
        st.warning("⚠️ No original image found. Please go back to the main page and upload an image first.")
        st.markdown("👉 [Go to Main Page](/)", unsafe_allow_html=True)
        return

    # Load the original image
    img_np = st.session_state.original_image
    st.image(img_np, caption="Original Image", use_container_width=True)

    # Load OCR model
    ocr = load_word()

    if st.button("🔍 Detect Words segmentation", type="primary"):
        with st.spinner("Detecting word regions..."):
            try:
                # Run OCR on the image (CPU) – we only need detection part
                result = ocr.ocr(img_np, cls=True)
                if not result or not result[0]:
                    st.warning("No words detected.")
                    return

                # Extract bounding boxes (ignore text and confidence)
                boxes = [item[0] for item in result[0]]

                # Optionally show the original image with boxes
                drawn_img = draw_word_boxes(img_np, boxes)
                st.image(drawn_img, caption="Word Detection Result (with bounding boxes)", use_container_width=True)

                # Crop each word
                cropped_words = crop_word_images(img_np, boxes)

                if not cropped_words:
                    st.warning("No valid word crops could be extracted.")
                    return

                # Display cropped word images in a grid
                st.subheader("✂️ Cropped Word Images")
                cols = st.columns(5)  # Display 5 per row
                for idx, (word_idx, word_img) in enumerate(cropped_words):
                    col = cols[idx % 5]
                    with col:
                        st.image(word_img, caption=f"Word {word_idx+1}", use_container_width=True)

                # Download all cropped words as ZIP
                st.subheader("📦 Download All Cropped Words")
                zip_buffer = create_zip_from_word_images(cropped_words)
                st.download_button(
                    label="Download as ZIP",
                    data=zip_buffer,
                    file_name=f"word_crops_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )

            except Exception as e:
                st.error(f"Error during word detection: {e}")
                st.error(traceback.format_exc())

    # Navigation back to main page
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Main Page"):
            st.switch_page("app.py")
    with col2:
        # ✅ الزر الجديد للانتقال إلى صفحة النتائج
        if st.button("📄 Show Extracted Arabic Text (Result Page)", type="primary"):
            st.switch_page("pages/result.py")

if __name__ == "__main__":
    main()