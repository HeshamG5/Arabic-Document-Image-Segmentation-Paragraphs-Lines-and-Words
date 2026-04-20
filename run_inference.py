import torch
import cv2
import yaml
from pathlib import Path
from src.models.unet import UNet
from src.evaluation.inference import predict_image, predict_folder  # تأكد من وجود هذا الاستيراد

def main():
    # الإعدادات
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # تحميل تكوين النموذج
    with open('configs/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # إنشاء النموذج
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        features=config['model']['features']
    ).to(device)

    # تحميل الأوزان المدربة
    checkpoint_path = 'experiments/exp_001/checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    # اختيار الوضع: إما صورة واحدة أو مجلد
    mode = 'folder'  # غيّر إلى 'single' لو أردت صورة واحدة

    if mode == 'single':
        # صورة واحدة
        image_path = 'data/raw/images/00116BC.png'
        mask = predict_image(
            model=model,
            
            image_path=image_path,
            device=device,
            image_size=(config['data']['image_height'], config['data']['image_width']),
            threshold=0.5,
            return_original_size=True
        )
        cv2.imwrite('test_mask.png', mask * 255)
        print("Saved mask to test_mask.png")

    elif mode == 'folder':
        # مجلد كامل
        predict_folder(
            model=model,
            input_folder='data/raw/images',          # ضع المسار المناسب
            output_folder='inference_results',
            device=device,
            image_size=(config['data']['image_height'], config['data']['image_width']),
            threshold=0.5,
            extract_lines=True,
            save_visualization=True
        )
        print("Inference on folder completed. Results saved in 'inference_results'.")

if __name__ == "__main__":
    main()