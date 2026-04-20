import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple

def read_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    قراءة صورة وإرجاعها بصيغة RGB.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    تغيير حجم الصورة إلى (العرض, الارتفاع) مع الحفاظ على نسبة الأبعاد إذا أردت،
    لكن هنا نستخدم resize مباشر.
    target_size: (height, width) أو (width, height)? سأفترض (width, height) لتوافق مع cv2.resize.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)