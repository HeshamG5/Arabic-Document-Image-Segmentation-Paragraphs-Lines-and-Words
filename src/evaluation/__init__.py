# يجعل المجلد evaluation حزمة Python
from .evaluate import evaluate_model
from .inference import predict_image, predict_folder

__all__ = ['evaluate_model', 'predict_image', 'predict_folder']