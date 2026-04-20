import cv2
import pytesseract
import numpy as np
import os

def preprocess_image(image_path):
    """
    تحميل الصورة ومعالجتها مسبقاً لتحسين دقة OCR.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"لم يتم العثور على الصورة في المسار: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    denoised = cv2.medianBlur(binary, 3)
    return img, denoised

def draw_bounding_boxes(image, data):
    """
    رسم الإطارات المحيطة حول النص المستخرج.
    """
    img_copy = image.copy()
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]
        text = data['text'][i].strip()
        if w > 0 and h > 0 and text:  # فقط إذا كان هناك نص وحجم موجب
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_copy, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    return img_copy

def arabic_segmentation(image_path, output_image_path="output_segmented.png"):
    """
    الوظيفة الرئيسية: تستقبل مسار الصورة وتقوم بتجزئة النص العربي.
    """
    if not os.path.exists(image_path):
        print(f"خطأ: الملف '{image_path}' غير موجود. تأكد من صحة المسار.")
        return None, None

    original_img, processed_img = preprocess_image(image_path)
    # إعدادات Tesseract: استخدام اللغة العربية مع PSM 6 (كتلة نصية واحدة)
    custom_config = r'--oem 3 --psm 6 -l ara'

    try:
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        data = pytesseract.image_to_data(processed_img, config=custom_config, output_type=pytesseract.Output.DICT)

        output_img = draw_bounding_boxes(original_img, data)
        cv2.imwrite(output_image_path, output_img)

        print("="*50)
        print("النص المستخرج من الصورة:")
        print(text.strip())
        print("="*50)
        print(f"\nتم حفظ صورة الإخراج مع التجزئة في: {os.path.abspath(output_image_path)}")
        return text, output_img

    except pytesseract.TesseractNotFoundError:
        print("خطأ: لم يتم العثور على Tesseract. تأكد من تثبيته وإضافته إلى PATH.")
        return None, None
    except Exception as e:
        print(f"حدث خطأ أثناء OCR: {e}")
        return None, None

if __name__ == "__main__":
    # ⚠️ استخدم صورتك المحددة هنا (باستخدام raw string r'...')
    image_file = r"C:\Users\asus\Desktop\sample\df1e16628e9a3adb4d0ae91d7ee23c15971e9234b81da4319b25033f.jpg"
    output_file = "output_segmented.png"  # يمكنك تغيير مسار الحفظ إذا أردت

    print(f"جاري معالجة الصورة: {image_file}")
    extracted_text, output_image = arabic_segmentation(image_file, output_file)

    if output_image is not None:
        # عرض الصورة الناتجة
        cv2.imshow("Segmented Image", output_image)
        print("\nاضغط أي مفتاح على نافذة الصورة لإغلاقها...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()