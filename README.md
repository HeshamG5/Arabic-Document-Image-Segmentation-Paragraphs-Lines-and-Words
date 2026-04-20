# Arabic-Document-Image-Segmentation-Paragraphs-Lines-and-Words
# Arabic Document Image Segmentation: Paragraphs, Lines, and Words

## Overview

This project provides an end‑to‑end solution for segmenting Arabic document images into three hierarchical levels: **paragraphs**, **text lines**, and **individual words**. It is designed to handle both printed and handwritten Arabic scripts, making it suitable for historical document analysis, OCR preprocessing, and digital archiving.

The system combines a deep learning‑based pixel‑wise segmentation model (U‑Net) with a lightweight yet accurate word boundary detection module powered by either **PaddleOCR** or a **Conditional Random Field (CRF)** approach. All functionality is exposed through an intuitive **Streamlit** web interface, allowing users to upload images, visualise results, and download segmentation masks or bounding boxes without writing a single line of code.

## Key Features

- **Multi‑level segmentation** – Extracts paragraphs, lines, and words from a single input image.
- **Arabic‑optimised** – Trained specifically on Arabic document images (printed and handwritten variants).
- **Two word‑level strategies**:
  - *PaddleOCR* – Pre‑trained, high‑accuracy word boxes for printed Arabic.
  - *CRF* – A training‑free, mask‑based refinement for scenarios where OCR is undesirable.
- **Interactive UI** – Built with Streamlit: drag‑and‑drop image upload, adjustable overlay transparency, side‑by‑side comparison.
- **Exportable results** – Download segmentation masks (PNG) or bounding box coordinates (JSON) for further processing.
- **No external annotation required at inference** – The U‑Net model works directly on raw document scans.

## Model & Training

The core segmentation engine is a **U‑Net** architecture (with optional ResNet/EfficientNet encoder). It is trained to classify each pixel into one of four classes: background, paragraph, line, or word. Training data consists of fully annotated Arabic document images – you can use public datasets (e.g., KHATT, PATS, AHTID) or your own labelled collection.

- **Framework**: TensorFlow / PyTorch (adjust to your implementation)
- **Input size**: configurable (e.g., 512×512 or full‑image inference)
- **Loss function**: categorical cross‑entropy + dice coefficient for handling class imbalance

## Word‑Level Segmentation Details

### Option 1 – PaddleOCR
PaddleOCR’s detection module (DB or EAST) provides word‑level bounding boxes directly. This option works best for clean, printed Arabic documents.

### Option 2 – CRF (Conditional Random Fields)
When OCR is not desired (e.g., for handwritten text or when preserving original glyph shapes), the CRF method takes the line‑level mask from U‑Net and splits each line into words using a combination of vertical projection profiles and CRF smoothing. This approach is lightweight and does not require a separate OCR engine.

## Requirements

- Python 3.8+
- streamlit
- opencv-python
- numpy
- tensorflow (or pytorch)
- paddlepaddle + paddleocr (optional for PaddleOCR mode)
- scikit-image (for CRF post-processing)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HeshamG5/Arabic-Text-Segmentation.git
   cd Arabic-Text-Segmentation
