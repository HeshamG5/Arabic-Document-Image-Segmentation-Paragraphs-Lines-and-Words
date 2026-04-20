import matplotlib.pyplot as plt
from pathlib import Path
import cv2

def visualize_sample(image_path: Path, mask_path: Path, save_path: Path = None):
    """عرض الصورة والماسك معًا."""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

if __name__ == "__main__":
    # مثال للاستخدام
    base = Path("data/processed/val")
    img_file = base / "images" / "001.png"
    mask_file = base / "masks" / "001.png"
    if img_file.exists() and mask_file.exists():
        visualize_sample(img_file, mask_file)