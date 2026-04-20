import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from collections import Counter
import re
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import cv2
import arabic_reshaper
from bidi.algorithm import get_display
import traceback

# =====================================================
# 0. إصلاح مشكلة PaddleOCR (اختياري)
# =====================================================
def apply_fix():
    try:
        import paddle.base.libpaddle as lp
        if hasattr(lp, 'AnalysisConfig'):
            lp.AnalysisConfig.set_optimization_level = lambda self, x: None
            lp.AnalysisConfig.disable_mkldnn = lambda self: None
    except:
        pass

apply_fix()
from paddleocr import PaddleOCR

# =====================================================
# 1. إعداد الصفحة والمتغيرات
# =====================================================
st.set_page_config(
    page_title="التحليل الذكي المتقدم | ARCHIVE AI PRO",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# --- إعداد Groq Client ---
client = Groq(api_key="os.environ.get('GROQ_API_KEY', 'your-key-here')")

def ask_groq(prompt, text_content):
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": """أنت خبير في تحليل البيانات واستخراج المعلومات. 
                عند طلب استخراج فورم، قم بالرد بصيغة JSON واضحة أو جدول منظم فقط. 
                عند طلب التحليل القانوني أو الطبي، كن دقيقاً جداً."""},
                {"role": "user", "content": f"{prompt}:\n\n{text_content}"}
            ],
            temperature=0.3,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"خطأ في الاتصال بالذكاء الاصطناعي: {e}"

# =====================================================
# 2. تحميل نموذج OCR (مخبأ)
# =====================================================
@st.cache_resource
def load_ocr():
    return PaddleOCR(lang='ar', use_textline_orientation=True, show_log=False, use_gpu=False)

# =====================================================
# 3. دالة لترتيب المربعات (سطراً سطراً)
# =====================================================
def sort_boxes(boxes):
    if not boxes:
        return []
    # ترتيب حسب الإحداثي Y (الصف)
    boxes.sort(key=lambda x: x[0][0][1])
    lines, current_line = [], [boxes[0]]
    for i in range(1, len(boxes)):
        if abs(boxes[i][0][0][1] - current_line[-1][0][0][1]) < 25:
            current_line.append(boxes[i])
        else:
            current_line.sort(key=lambda x: x[0][0][0], reverse=True)  # من اليمين لليسار
            lines.extend(current_line)
            current_line = [boxes[i]]
    current_line.sort(key=lambda x: x[0][0][0], reverse=True)
    lines.extend(current_line)
    return lines

# =====================================================
# 4. نظام تحليل الثقة المتقدم (نفس الكود الأصلي)
# =====================================================
class AdvancedConfidenceAnalyzer:
    def __init__(self):
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        self.common_arabic_words = self.load_common_arabic_words()

    def load_common_arabic_words(self):
        return set([
            'في', 'من', 'إلى', 'على', 'كان', 'هذا', 'هذه', 'قال', 'وقال', 'كانت',
            'ذلك', 'الله', 'فيها', 'عليه', 'عليها', 'عند', 'مع', 'أنا', 'أنت', 'نحن',
            'هم', 'هن', 'هو', 'هي', 'و', 'ف', 'ب', 'ل', 'ك', 'لا', 'لم', 'لن',
            'إن', 'أن', 'إذا', 'حين', 'بعد', 'قبل', 'فوق', 'تحت', 'بين', 'خلال',
            'يوم', 'شهر', 'سنة', 'وقت', 'الآن', 'هنا', 'هناك', 'كل', 'بعض', 'أي',
            'الذي', 'التي', 'الذين', 'اللواتي', 'اللائي', 'اللذان', 'اللتان',
            'كانوا', 'كانوا', 'كانت', 'كن', 'يكون', 'تكون', 'نكون', 'أكون',
            'ولم', 'ولن', 'وسوف', 'سوف', 'قد', 'لقد', 'هل', 'أ', 'ماذا', 'لماذا',
            'كيف', 'أين', 'متى', 'كم', 'أي', 'منذ', 'حتى', 'عندما', 'حيث', 'إذ'
        ])

    def analyze_confidence(self, text, ocr_results=None, image=None):
        words = text.split()
        detailed_analysis = []
        suspicious_regions = []

        for i, word in enumerate(words):
            word_analysis = self.analyze_word_confidence(word, i, words)
            detailed_analysis.append(word_analysis)
            if word_analysis['confidence'] < 70:
                suspicious_regions.append({
                    'word': word,
                    'position': i,
                    'confidence': word_analysis['confidence'],
                    'reasons': word_analysis['reasons'],
                    'severity': 'high' if word_analysis['confidence'] < 50 else 'medium'
                })

        confidence_scores = [w['confidence'] for w in detailed_analysis]
        overall_stats = {
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'min_confidence': np.min(confidence_scores) if confidence_scores else 0,
            'max_confidence': np.max(confidence_scores) if confidence_scores else 0,
            'std_confidence': np.std(confidence_scores) if confidence_scores else 0,
            'high_confidence_count': len([c for c in confidence_scores if c >= 80]),
            'medium_confidence_count': len([c for c in confidence_scores if 50 <= c < 80]),
            'low_confidence_count': len([c for c in confidence_scores if c < 50]),
            'suspicious_count': len(suspicious_regions)
        }

        highlighted_image = None
        if image is not None and ocr_results is not None:
            highlighted_image = self.highlight_suspicious_regions(image, ocr_results, detailed_analysis)

        return {
            'word_analysis': detailed_analysis,
            'overall_stats': overall_stats,
            'suspicious_regions': suspicious_regions,
            'highlighted_image': highlighted_image,
            'text': text
        }

    def analyze_word_confidence(self, word, position, all_words):
        confidence = 100.0
        reasons = []
        details = {}

        # تحليل الطول
        length_score, length_reason, length_details = self.analyze_length(word)
        confidence += length_score
        if length_reason: reasons.append(length_reason)
        if length_details: details.update(length_details)

        # تحليل التركيب الحرفي
        char_score, char_reason, char_details = self.analyze_character_structure(word)
        confidence += char_score
        if char_reason: reasons.append(char_reason)
        if char_details: details.update(char_details)

        # تحليل نسبة العربية
        arabic_score, arabic_reason, arabic_details = self.analyze_arabic_ratio(word)
        confidence += arabic_score
        if arabic_reason: reasons.append(arabic_reason)
        if arabic_details: details.update(arabic_details)

        # تحليل النمط اللغوي
        pattern_score, pattern_reason, pattern_details = self.analyze_language_pattern(word)
        confidence += pattern_score
        if pattern_reason: reasons.append(pattern_reason)
        if pattern_details: details.update(pattern_details)

        # تحليل السياق
        context_score, context_reason, context_details = self.analyze_context(word, position, all_words)
        confidence += context_score
        if context_reason: reasons.append(context_reason)
        if context_details: details.update(context_details)

        # تحليل القاموس
        dict_score, dict_reason, dict_details = self.check_dictionary(word)
        confidence += dict_score
        if dict_reason: reasons.append(dict_reason)
        if dict_details: details.update(dict_details)

        # تحليل التكرار
        rep_score, rep_reason, rep_details = self.analyze_repetition(word)
        confidence += rep_score
        if rep_reason: reasons.append(rep_reason)
        if rep_details: details.update(rep_details)

        # تحليل علامات الترقيم
        punct_score, punct_reason, punct_details = self.analyze_punctuation(word)
        confidence += punct_score
        if punct_reason: reasons.append(punct_reason)
        if punct_details: details.update(punct_details)

        confidence = max(0, min(100, round(confidence, 1)))
        if confidence >= 80:
            level, level_color = "عالية", "#2ecc71"
        elif confidence >= 60:
            level, level_color = "متوسطة", "#f39c12"
        elif confidence >= 40:
            level, level_color = "منخفضة", "#e67e22"
        else:
            level, level_color = "منخفضة جداً", "#e74c3c"

        return {
            'word': word,
            'position': position,
            'confidence': confidence,
            'level': level,
            'level_color': level_color,
            'reasons': reasons,
            'details': details,
            'length': len(word)
        }

    def analyze_length(self, word):
        score = 0
        reason = None
        details = {}
        if len(word) == 0:
            score = -100
            reason = "كلمة فارغة"
        elif len(word) == 1:
            if word in ['و', 'ف', 'ب', 'ل', 'ك']:
                details['length_type'] = 'حرف عطف/جر مقبول'
            else:
                score = -30
                reason = "حرف منفرد غير شائع"
                details['length_type'] = 'حرف منفرد مشبوه'
        elif len(word) > 20:
            score = -40
            reason = "كلمة طويلة جداً (احتمال دمج كلمتين)"
            details['length_type'] = 'طويلة جداً'
        elif len(word) > 15:
            score = -20
            reason = "كلمة طويلة (تحتاج تدقيق)"
            details['length_type'] = 'طويلة'
        elif len(word) < 3:
            score = -10
            reason = "كلمة قصيرة غير شائعة"
            details['length_type'] = 'قصيرة'
        else:
            details['length_type'] = 'طبيعية'
        details['length'] = len(word)
        return score, reason, details

    def analyze_character_structure(self, word):
        score = 0
        reason = None
        details = {}
        for i in range(len(word)-2):
            if word[i] == word[i+1] == word[i+2]:
                score -= 20
                reason = f"تكرار غير طبيعي للحرف '{word[i]}'"
                details['repetition'] = word[i:i+3]
                break
        rare_letters = 'ثخذضظغ'
        for l in rare_letters:
            if l in word and word.count(l) > 2:
                score -= 10
                reason = f"تكرار مفرط لحرف نادر '{l}'"
                details['rare_letter'] = l
                break
        return score, reason, details

    def analyze_arabic_ratio(self, word):
        score = 0
        reason = None
        details = {}
        arabic_chars = len(self.arabic_pattern.findall(word))
        total_chars = len(word)
        if total_chars > 0:
            ratio = arabic_chars / total_chars
            details['arabic_ratio'] = round(ratio * 100, 1)
            if ratio < 0.3:
                score = -50
                reason = "نسبة منخفضة جداً من الحروف العربية"
            elif ratio < 0.5:
                score = -35
                reason = "أقل من 50% حروف عربية"
            elif ratio < 0.7:
                score = -20
                reason = "نسبة منخفضة من الحروف العربية"
            elif ratio < 0.9:
                score = -5
                reason = "نسبة مقبولة من الحروف العربية"
            else:
                score = 5
        return score, reason, details

    def analyze_language_pattern(self, word):
        score = 0
        reason = None
        details = {}
        if re.search(r'\d', word) and not word.isdigit():
            score -= 25
            reason = "وجود أرقام داخل الكلمة"
            details['contains_numbers'] = re.findall(r'\d+', word)
        if re.search(r'[a-zA-Z]', word):
            score -= 30
            reason = "وجود أحرف إنجليزية"
            details['contains_english'] = re.findall(r'[a-zA-Z]+', word)
        unusual_punct = re.findall(r'[@#$%^&*_+=|\\<>~`]', word)
        if unusual_punct:
            score -= 20
            reason = "وجود علامات ترقيم غير عادية"
            details['unusual_punctuation'] = unusual_punct
        return score, reason, details

    def analyze_context(self, word, position, all_words):
        score = 0
        reason = None
        details = {}
        if len(all_words) > 1:
            if position > 0:
                prev_word = all_words[position - 1]
                details['prev_word'] = prev_word
                if prev_word in self.common_arabic_words and word not in self.common_arabic_words and len(word) < 3:
                    score -= 15
                    reason = "كلمة قصيرة غير معروفة بعد كلمة شائعة"
            if position < len(all_words) - 1:
                next_word = all_words[position + 1]
                details['next_word'] = next_word
                if len(word) > 15 and len(next_word) > 15:
                    score -= 20
                    reason = "كلمتين طويلتين جداً متتاليتين"
        return score, reason, details

    def check_dictionary(self, word):
        score = 0
        reason = None
        details = {}
        if word in self.common_arabic_words:
            score = 20
            details['dictionary_status'] = 'معروفة'
        elif len(word) > 3 and word[1:] in self.common_arabic_words:
            score = 10
            reason = "كلمة معروفة مع حرف إضافي محتمل"
            details['dictionary_status'] = 'جزئية'
            details['suggested_word'] = word[1:]
        else:
            details['dictionary_status'] = 'غير معروفة'
        return score, reason, details

    def analyze_repetition(self, word):
        score = 0
        reason = None
        details = {}
        if len(word) > 3 and word[:2] == word[2:4]:
            score -= 20
            reason = "تكرار مقطع من الكلمة"
            details['repetition_type'] = 'syllable_repetition'
        if len(word) > 3 and word[0] == word[-1] and word[0] not in ['و', 'ف', 'ب']:
            score -= 10
            reason = "كلمة تبدأ وتنتهي بنفس الحرف (غير شائع)"
        return score, reason, details

    def analyze_punctuation(self, word):
        score = 0
        reason = None
        details = {}
        if '.' in word and len(word) > 1:
            score -= 15
            reason = "وجود نقطة داخل الكلمة"
        elif ',' in word and len(word) > 1:
            score -= 10
            reason = "وجود فاصلة داخل الكلمة"
        return score, reason, details

    def highlight_suspicious_regions(self, image, ocr_results, word_analysis):
        if image is None or ocr_results is None:
            return image
        img = image.copy()
        draw = ImageDraw.Draw(img, 'RGBA')
        colors = {
            'high': (46, 204, 113, 50),
            'medium': (241, 196, 15, 50),
            'low': (230, 126, 34, 50),
            'very_low': (231, 76, 60, 50)
        }
        for i, (result, analysis) in enumerate(zip(ocr_results[0], word_analysis)):
            box = result[0]
            conf = analysis['confidence']
            if conf >= 80:
                color, border = colors['high'], '#2ecc71'
            elif conf >= 60:
                color, border = colors['medium'], '#f1c40f'
            elif conf >= 40:
                color, border = colors['low'], '#e67e22'
            else:
                color, border = colors['very_low'], '#e74c3c'
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            draw.rectangle([(min(xs), min(ys)), (max(xs), max(ys))], outline=border, fill=color, width=3)
            if conf < 70:
                draw.text((min(xs), min(ys)-20), f"{conf}%", fill=border)
        return img

# =====================================================
# 5. نظام تقييم الدقة المتقدم (مختصر)
# =====================================================
class AdvancedAccuracyEvaluator:
    def __init__(self):
        self.confidence_analyzer = AdvancedConfidenceAnalyzer()

    def evaluate_accuracy(self, text, ocr_results=None, image=None):
        confidence_results = self.confidence_analyzer.analyze_confidence(text, ocr_results, image)
        accuracy_metrics = self.calculate_accuracy_metrics(text, confidence_results)
        error_analysis = self.analyze_potential_errors(text, confidence_results)
        return {
            'confidence_analysis': confidence_results,
            'accuracy_metrics': accuracy_metrics,
            'error_analysis': error_analysis,
            'overall_accuracy': self.calculate_overall_accuracy(accuracy_metrics, confidence_results)
        }

    def calculate_accuracy_metrics(self, text, confidence_results):
        word_analysis = confidence_results['word_analysis']
        distribution = {
            'high_precision_words': len([w for w in word_analysis if w['confidence'] >= 90]),
            'good_precision_words': len([w for w in word_analysis if 80 <= w['confidence'] < 90]),
            'medium_precision_words': len([w for w in word_analysis if 70 <= w['confidence'] < 80]),
            'low_precision_words': len([w for w in word_analysis if w['confidence'] < 70])
        }
        word_lengths = [len(w) for w in text.split()]
        normal_length_ratio = len([l for l in word_lengths if 3 <= l <= 10]) / len(word_lengths) if word_lengths else 0
        structural = {'normal_length_ratio': normal_length_ratio}
        context_accuracy = {'context_score': 0}
        dictionary = self.analyze_dictionary_accuracy(word_analysis)
        weighted_score = self.calculate_weighted_score(distribution, structural, context_accuracy, dictionary)
        return {
            'distribution': distribution,
            'structural': structural,
            'contextual': context_accuracy,
            'dictionary': dictionary,
            'weighted_score': weighted_score
        }

    def analyze_dictionary_accuracy(self, word_analysis):
        known = sum(1 for w in word_analysis if w['details'].get('dictionary_status') == 'معروفة')
        total = len(word_analysis)
        return {'known_ratio': (known / total * 100) if total else 0}

    def calculate_weighted_score(self, dist, struct, context, dictionary):
        total_words = sum(dist.values())
        if total_words == 0:
            return 0
        high_ratio = (dist['high_precision_words'] + dist['good_precision_words']) / total_words
        score = high_ratio * 100 * 0.3
        score += struct['normal_length_ratio'] * 100 * 0.25
        score += (100 + context['context_score']) * 0.25
        score += dictionary['known_ratio'] * 0.2
        return round(score, 1)

    def analyze_potential_errors(self, text, confidence_results):
        error_types = {'segmentation_errors': [], 'character_errors': [], 'spelling_errors': [], 'context_errors': []}
        for word_info in confidence_results['word_analysis']:
            reasons = word_info.get('reasons', [])
            word = word_info['word']
            for reason in reasons:
                if 'طويلة جداً' in reason or 'دمج' in reason:
                    error_types['segmentation_errors'].append({'word': word, 'confidence': word_info['confidence'], 'reason': reason})
                elif 'حرف' in reason and ('تكرار' in reason or 'نادر' in reason):
                    error_types['character_errors'].append({'word': word, 'confidence': word_info['confidence'], 'reason': reason})
                elif 'إنجليزية' in reason or 'أرقام' in reason or 'علامات' in reason:
                    error_types['character_errors'].append({'word': word, 'confidence': word_info['confidence'], 'reason': reason})
                elif 'قاموس' in reason or 'سياق' in reason:
                    error_types['spelling_errors'].append({'word': word, 'confidence': word_info['confidence'], 'reason': reason})
                elif 'سياق' in reason.lower():
                    error_types['context_errors'].append({'word': word, 'confidence': word_info['confidence'], 'reason': reason})
        error_stats = {k: {'count': len(v), 'examples': v[:5]} for k, v in error_types.items()}
        return {'error_types': error_types, 'error_stats': error_stats, 'total_errors': sum(len(e) for e in error_types.values())}

    def calculate_overall_accuracy(self, accuracy_metrics, confidence_results):
        scores = []
        scores.append(confidence_results['overall_stats']['avg_confidence'] * 0.4)
        scores.append(accuracy_metrics['weighted_score'] * 0.6)
        overall = np.mean(scores) if scores else 0
        grade = self.get_grade(overall)
        stars = '⭐' * int(overall/20) + '☆' * (5 - int(overall/20))
        level = self.get_accuracy_level(overall)
        return {'score': round(overall, 1), 'grade': grade, 'stars': stars, 'level': level}

    def get_grade(self, score):
        if score >= 95: return 'A+'
        elif score >= 90: return 'A'
        elif score >= 85: return 'B+'
        elif score >= 80: return 'B'
        elif score >= 75: return 'C+'
        elif score >= 70: return 'C'
        elif score >= 65: return 'D+'
        elif score >= 60: return 'D'
        else: return 'F'

    def get_accuracy_level(self, score):
        if score >= 90: return 'ممتازة'
        elif score >= 80: return 'جيدة جداً'
        elif score >= 70: return 'جيدة'
        elif score >= 60: return 'مقبولة'
        else: return 'ضعيفة'

def display_advanced_accuracy_report(accuracy_results):
    confidence = accuracy_results['confidence_analysis']
    metrics = accuracy_results['accuracy_metrics']
    errors = accuracy_results['error_analysis']
    overall = accuracy_results['overall_accuracy']

    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 25px; margin-bottom: 30px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div><h2 style="color: white; margin: 0;">🎯 الدقة الشاملة</h2>
                <p style="color: rgba(255,255,255,0.8); margin: 5px 0;">{overall['grade']} · {overall['level']}</p></div>
                <div style="text-align: center;">
                    <span style="color: white; font-size: 64px; font-weight: bold;">{overall['score']}%</span>
                    <p style="color: #ffd700; font-size: 24px; margin: 0;">{overall['stars']}</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div style="background: rgba(52,152,219,0.1); padding: 20px; border-radius: 15px; text-align: center;">
                <span style="color: #3498db; font-size: 30px;">📊</span>
                <p style="color: white;">متوسط الثقة</p>
                <h3 style="color: #3498db;">{confidence['overall_stats']['avg_confidence']:.1f}%</h3>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div style="background: rgba(46,204,113,0.1); padding: 20px; border-radius: 15px; text-align: center;">
                <span style="color: #2ecc71; font-size: 30px;">📈</span>
                <p style="color: white;">أعلى ثقة</p>
                <h3 style="color: #2ecc71;">{confidence['overall_stats']['max_confidence']:.1f}%</h3>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div style="background: rgba(230,126,34,0.1); padding: 20px; border-radius: 15px; text-align: center;">
                <span style="color: #e67e22; font-size: 30px;">📉</span>
                <p style="color: white;">أدنى ثقة</p>
                <h3 style="color: #e67e22;">{confidence['overall_stats']['min_confidence']:.1f}%</h3>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div style="background: rgba(155,89,182,0.1); padding: 20px; border-radius: 15px; text-align: center;">
                <span style="color: #9b59b6; font-size: 30px;">⚠️</span>
                <p style="color: white;">كلمات مشبوهة</p>
                <h3 style="color: #9b59b6;">{confidence['overall_stats']['suspicious_count']}</h3>
            </div>
        """, unsafe_allow_html=True)

    # توزيع الثقة
    fig = go.Figure(data=[go.Pie(
        labels=['عالية (80-100%)', 'متوسطة (60-80%)', 'منخفضة (40-60%)', 'منخفضة جداً (<40%)'],
        values=[
            confidence['overall_stats']['high_confidence_count'],
            confidence['overall_stats']['medium_confidence_count'],
            len([w for w in confidence['word_analysis'] if 40 <= w['confidence'] < 60]),
            len([w for w in confidence['word_analysis'] if w['confidence'] < 40])
        ],
        marker_colors=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c'],
        hole=0.4
    )])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 6. دوال الإحصائيات المتقدمة
# =====================================================
def create_advanced_statistics(text):
    text_clean = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    words = text_clean.split()
    chars = list(text_clean.replace(' ', ''))
    if not words:
        return None
    word_lengths = [len(w) for w in words]
    fig1 = px.histogram(word_lengths, title='📊 توزيع أطوال الكلمات', labels={'value': 'طول الكلمة', 'count': 'عدد الكلمات'},
                        color_discrete_sequence=['#e67e22'], nbins=15)
    fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')

    if chars:
        arabic_letters = 'ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئة'
        letter_counts = []
        letters_list = []
        for l in arabic_letters:
            cnt = chars.count(l)
            if cnt > 0:
                try:
                    reshaped = arabic_reshaper.reshape(l)
                    letters_list.append(get_display(reshaped))
                except:
                    letters_list.append(l)
                letter_counts.append(cnt)
        fig2 = go.Figure(data=[go.Bar(x=letters_list, y=letter_counts, marker_color='#2ecc71')])
        fig2.update_layout(title='🔤 توزيع الحروف العربية', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    else:
        fig2 = None

    unique_words = len(set(words))
    total_words = len(words)
    total_chars = len(chars)
    avg_word_length = np.mean(word_lengths) if word_lengths else 0
    lexical_diversity = (unique_words / total_words * 100) if total_words else 0
    longest_word = max(words, key=len) if words else ""
    try:
        longest_word_display = get_display(arabic_reshaper.reshape(longest_word))
    except:
        longest_word_display = longest_word

    short_words = len([w for w in word_lengths if w <= 3])
    medium_words = len([w for w in word_lengths if 4 <= w <= 6])
    long_words = len([w for w in word_lengths if w >= 7])
    fig3 = go.Figure(data=[go.Pie(labels=['قصيرة', 'متوسطة', 'طويلة'], values=[short_words, medium_words, long_words],
                                  marker_colors=['#3498db', '#2ecc71', '#e67e22'], hole=0.4)])
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')

    stats = {
        'total_words': total_words, 'unique_words': unique_words, 'total_chars': total_chars,
        'avg_word_length': round(avg_word_length, 2), 'lexical_diversity': round(lexical_diversity, 2),
        'longest_word': longest_word_display, 'longest_word_len': len(longest_word)
    }
    return {'figures': [fig1, fig2, fig3], 'statistics': stats}

def display_stats_cards(stats):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 25px; border-radius: 20px; text-align: center;">
                <h3 style="color: white;">📝 إجمالي الكلمات</h3>
                <p style="color: white; font-size: 48px; font-weight: bold;">{stats['total_words']}</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb, #f5576c); padding: 25px; border-radius: 20px; text-align: center;">
                <h3 style="color: white;">🆕 كلمات فريدة</h3>
                <p style="color: white; font-size: 48px; font-weight: bold;">{stats['unique_words']}</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #5f2c82, #49a09d); padding: 25px; border-radius: 20px; text-align: center;">
                <h3 style="color: white;">📊 التنوع اللغوي</h3>
                <p style="color: white; font-size: 48px; font-weight: bold;">{stats['lexical_diversity']}%</p>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fc4a1a, #f7b733); padding: 25px; border-radius: 20px; text-align: center;">
                <h3 style="color: white;">🔤 متوسط طول الكلمة</h3>
                <p style="color: white; font-size: 48px; font-weight: bold;">{stats['avg_word_length']}</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #42275a, #734b6d); padding: 25px; border-radius: 20px; text-align: center;">
                <h4 style="color: white;">🏆 أطول كلمة</h4>
                <p style="color: #ffd700; font-size: 32px; font-weight: bold;">{stats['longest_word'][::-1]}</p>
                <div style="background: rgba(255,255,255,0.1); padding: 5px; border-radius: 10px;">{stats['longest_word_len']} حرف</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1d976c, #93f9b9); padding: 25px; border-radius: 20px; text-align: center;">
                <h4 style="color: white;">📈 إجمالي الحروف</h4>
                <p style="color: white; font-size: 48px; font-weight: bold;">{stats['total_chars']}</p>
            </div>
        """, unsafe_allow_html=True)

# =====================================================
# 7. CSS والهيدر
# =====================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap');
    * { font-family: 'Cairo', sans-serif; }
    .stApp { background: radial-gradient(circle at 0% 0%, #1a2634 0%, #0f172a 100%); color: #ffffff; }
    .hero-section {
        background: linear-gradient(135deg, rgba(230,126,34,0.1), rgba(243,156,18,0.1));
        padding: 30px; border-radius: 30px; margin-bottom: 30px;
        border: 1px solid rgba(230,126,34,0.3); backdrop-filter: blur(10px); text-align: center;
    }
    .hero-title {
        font-size: 48px; font-weight: 900;
        background: linear-gradient(135deg, #e67e22, #f39c12);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .main-box, .ai-box {
        background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
        padding: 25px; border-radius: 25px; border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        border-radius: 50px; background: linear-gradient(135deg, #e67e22, #f39c12);
        color: white; font-weight: bold; border: none; padding: 12px 30px;
        transition: 0.3s;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(230,126,34,0.4); }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05); border-radius: 50px; padding: 10px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e67e22, #f39c12) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🚀 ARCHIVE AI PRO • التحليل الذكي المتقدم</h1>
        <p class="hero-subtitle" style="color:#94a3b8;">نظام متكامل لتحليل واستخراج المعلومات من الوثائق العربية</p>
    </div>
""", unsafe_allow_html=True)

# =====================================================
# 8. المعالجة الأساسية: OCR على الصورة الأصلية
# =====================================================
if 'original_image' not in st.session_state or st.session_state.original_image is None:
    st.warning("⚠️ لم يتم العثور على صورة. الرجاء العودة إلى الصفحة الرئيسية ورفع صورة أولاً.")
    if st.button("🏠 العودة إلى الصفحة الرئيسية"):
        st.switch_page("app.py")
    st.stop()

original_img = st.session_state.original_image  # numpy array (RGB)
st.subheader("🖼️ الصورة المدخلة")
st.image(original_img, use_container_width=True)

# تنفيذ OCR إذا لم تكن النتائج موجودة مسبقاً في الجلسة
if 'full_text' not in st.session_state or 'viz_img' not in st.session_state:
    with st.spinner("⏳ جاري تحليل الصورة واستخراج النصوص باستخدام PaddleOCR..."):
        try:
            ocr = load_ocr()
            result = ocr.ocr(original_img, cls=True)
            if not result or not result[0]:
                st.error("❌ لم يتم العثور على أي نص في الصورة.")
                st.stop()
            
            # ترتيب المربعات حسب الأسطر
            sorted_boxes = sort_boxes(result[0])
            
            # رسم المربعات على نسخة من الصورة
            viz_img = Image.fromarray(original_img).copy()
            draw = ImageDraw.Draw(viz_img)
            full_text = ""
            chars_text = ""
            
            for item in sorted_boxes:
                box, (text, score) = item[0], item[1]
                # عكس النص (لأن PaddleOCR يعيده بالترتيب الصحيح لكن التطبيق الأصلي يعكسه)
                corrected = text[::-1]
                full_text += f"{corrected}\n"
                chars_text += f"{' '.join(list(corrected))}\n"
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                draw.rectangle([(min(xs), min(ys)), (max(xs), max(ys))], outline="#39FF14", width=3)
            
            st.session_state['full_text'] = full_text
            st.session_state['chars_text'] = chars_text
            st.session_state['viz_img'] = viz_img
            st.session_state['ocr_results'] = result  # حفظ النتائج الخام للتحليلات المتقدمة
            st.success("✅ تم استخراج النص بنجاح!")
            st.rerun()
        except Exception as e:
            st.error(f"خطأ في OCR: {str(e)}")
            st.error(traceback.format_exc())
            st.stop()

# =====================================================
# 9. عرض النتائج والتحليلات (مثل الكود الأصلي)
# =====================================================
full_text = st.session_state['full_text']
viz_img = st.session_state['viz_img']
chars_text = st.session_state['chars_text']

# إحصائيات سريعة
line_count = len([line for line in full_text.splitlines() if line.strip()])
char_count = len(full_text.replace("\n", "").replace(" ", ""))
progress_bar = st.progress(0)
for i in range(100):
    time.sleep(0.01)
    progress_bar.progress(i + 1)
time.sleep(0.2)
progress_bar.empty()

st.markdown(f"""
    <div class="timestamp" style="color:#94a3b8; font-size:14px; margin-bottom:20px;">
        🕒 آخر تحديث: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
""", unsafe_allow_html=True)

col_img, col_txt = st.columns([1, 1.2], gap="large")

with col_img:
    st.markdown('<div class="main-box">', unsafe_allow_html=True)
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <span style="font-size: 40px;">🖼️</span>
            <h2 style="color: white; margin: 0;">تحليل التجزئة الهيكلية</h2>
        </div>
    """, unsafe_allow_html=True)
    st.image(viz_img, use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
            <div style="background: rgba(46,204,113,0.1); padding: 10px; border-radius: 10px; text-align: center;">
                <span style="color:#2ecc71;">📊</span>
                <p style="color:white;">{line_count} سطر</p>
            </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
            <div style="background: rgba(230,126,34,0.1); padding: 10px; border-radius: 10px; text-align: center;">
                <span style="color:#e67e22;">🔤</span>
                <p style="color:white;">{char_count} حرف</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # خيارات الذكاء الاصطناعي
    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(230,126,34,0.1), rgba(243,156,18,0.1));
                    padding: 20px; border-radius: 20px; margin-top: 20px;">
            <h3 style="color: white; text-align: center;">🧬 وظائف المعالجة المتقدمة</h3>
        </div>
    """, unsafe_allow_html=True)
    option = st.selectbox("", [
        "استخراج فورم (اسم، تاريخ، أرقام)", "تحويل النص إلى جدول بيانات",
        "تصنيف محتوى الوثيقة", "تحليل الكلمات المفتاحية والأرشفة",
        "تدقيق لغوي وتصحيح OCR", "ترجمة احترافية (English)", "تلخيص تنفيذي"
    ], label_visibility="collapsed")
    if st.button("🚀 تنفيذ المعالجة المتقدمة", use_container_width=True):
        with st.spinner("⏳ جاري تحليل البيانات باستخدام الذكاء الاصطناعي..."):
            prompts = {
                "استخراج فورم (اسم، تاريخ، أرقام)": "استخرج المعلومات الشخصية (الاسم، التاريخ، العناوين، الأرقام) من النص بصيغة JSON واضحة للاستخدام في قاعدة بيانات",
                "تحويل النص إلى جدول بيانات": "قم بتحويل المعلومات الموجودة في النص إلى جدول Markdown منظم يحتوي على أعمدة وقيم",
                "تصنيف محتوى الوثيقة": "حلل النص وحدد نوع الوثيقة (فاتورة، رسالة، عقد، هوية) مع ذكر الأسباب",
                "تحليل الكلمات المفتاحية والأرشفة": "استخرج أهم 10 كلمات مفتاحية للأرشفة واقترح اسماً مناسباً للملف",
                "تدقيق لغوي وتصحيح OCR": "صحح أخطاء الـ OCR وحسن جودة النص لغوياً",
                "ترجمة احترافية (English)": "Translate the following text to professional English accurately",
                "تلخيص تنفيذي": "لخص محتوى الوثيقة في سطرين فقط بشكل تنفيذي"
            }
            result = ask_groq(prompts[option], full_text)
            st.session_state['ai_res'] = result
            st.balloons()

with col_txt:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📄 النص الخام", "🔠 تحليل الحروف", "📊 إحصائيات متقدمة",
        "🎯 تحليل الثقة", "📋 تقييم متقدم", "🚀 المخرجات الذكية"
    ])
    with tab1:
        st.markdown('<div class="main-box">', unsafe_allow_html=True)
        st.text_area("", full_text, height=450, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    with tab2:
        st.markdown('<div class="main-box">', unsafe_allow_html=True)
        st.text_area("", chars_text, height=450, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    with tab3:
        st.markdown('<div class="main-box">', unsafe_allow_html=True)
        stats_results = create_advanced_statistics(full_text)
        if stats_results:
            display_stats_cards(stats_results['statistics'])
            st.plotly_chart(stats_results['figures'][0], use_container_width=True)
            if stats_results['figures'][1]:
                st.plotly_chart(stats_results['figures'][1], use_container_width=True)
            st.plotly_chart(stats_results['figures'][2], use_container_width=True)
        else:
            st.warning("⚠️ لا توجد كلمات كافية للتحليل")
        st.markdown('</div>', unsafe_allow_html=True)
    with tab4:
        st.markdown('<div class="main-box">', unsafe_allow_html=True)
        confidence_analyzer = AdvancedConfidenceAnalyzer()
        ocr_results = st.session_state.get('ocr_results', None)
        confidence_results = confidence_analyzer.analyze_confidence(
            full_text, ocr_results, st.session_state.get('viz_img', None)
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("متوسط الثقة", f"{confidence_results['overall_stats']['avg_confidence']:.1f}%")
        with col2:
            st.metric("كلمات موثوقة", confidence_results['overall_stats']['high_confidence_count'])
        with col3:
            st.metric("كلمات مشبوهة", confidence_results['overall_stats']['suspicious_count'])
        if confidence_results['suspicious_regions']:
            with st.expander("🔍 الكلمات المشبوهة (أقل من 70% ثقة)"):
                for r in confidence_results['suspicious_regions'][:10]:
                    st.markdown(f"- **{r['word']}** ({r['confidence']}%) - {', '.join(r['reasons'][:2])}")
        if confidence_results['highlighted_image']:
            st.image(confidence_results['highlighted_image'], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with tab5:
        st.markdown('<div class="main-box">', unsafe_allow_html=True)
        evaluator = AdvancedAccuracyEvaluator()
        acc_results = evaluator.evaluate_accuracy(full_text, ocr_results, st.session_state.get('viz_img', None))
        display_advanced_accuracy_report(acc_results)
        st.markdown('</div>', unsafe_allow_html=True)
    with tab6:
        st.markdown('<div class="ai-box">', unsafe_allow_html=True)
        ai_output = st.session_state.get('ai_res', "⚠️ لم يتم تنفيذ أي مهمة بعد. اختر مهمة من القائمة واضغط على 'تنفيذ المعالجة المتقدمة'")
        st.markdown(ai_output)
        if 'ai_res' in st.session_state:
            st.download_button("📥 تحميل النتائج", ai_output, file_name=f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        st.markdown('</div>', unsafe_allow_html=True)

st.divider()
if st.button("🔄 العودة إلى الصفحة الرئيسية", use_container_width=False):
    st.switch_page("app.py")
