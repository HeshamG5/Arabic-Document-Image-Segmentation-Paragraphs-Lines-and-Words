import json
from typing import List, Tuple, Optional

def load_json(json_path: str) -> dict:
    """تحميل ملف JSON وإرجاعه كقاموس."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_page_polygon(annotation: dict) -> Optional[List[Tuple[int, int]]]:
    """استخراج نقاط مضلع الصفحة من التعليق."""
    for obj in annotation.get('objects', []):
        if obj.get('classTitle') == 'Page':
            return obj['points']['exterior']
    return None

def get_text_lines_bboxes(annotation: dict) -> List[Tuple[int, int, int, int]]:
    """
    استخراج جميع مستطيلات السطور النصية (Body text و Title).
    الإرجاع: قائمة من (x_min, y_min, x_max, y_max)
    """
    bboxes = []
    for obj in annotation.get('objects', []):
        cls = obj.get('classTitle')
        if cls not in ['Body text', 'Title']:
            continue
        exterior = obj['points']['exterior']
        if len(exterior) != 2:
            continue  # نتأكد أنه مستطيل
        x1, y1 = exterior[0]
        x2, y2 = exterior[1]
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes

def get_text_lines_with_text(annotation: dict) -> List[dict]:
    """
    استخراج السطور مع النص إذا كان متوفراً (للملفات التي تحتوي Transcription).
    """
    lines = []
    for obj in annotation.get('objects', []):
        cls = obj.get('classTitle')
        if cls not in ['Body text', 'Title']:
            continue
        exterior = obj['points']['exterior']
        if len(exterior) != 2:
            continue
        x1, y1 = exterior[0]
        x2, y2 = exterior[1]
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        text = ''
        for tag in obj.get('tags', []):
            if tag.get('name') == 'Transcription':
                text = tag.get('value', '')
                break
        lines.append({
            'bbox': (x_min, y_min, x_max, y_max),
            'text': text,
            'class': cls
        })
    return lines