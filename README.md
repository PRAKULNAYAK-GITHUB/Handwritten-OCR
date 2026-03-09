---

# 2️⃣ Handwritten OCR using CRNN

```markdown
# Handwritten OCR using CRNN

## Overview
This project implements a deep learning based handwritten word recognition system using a Convolutional Recurrent Neural Network (CRNN). The system combines CNN-based feature extraction with recurrent sequence modeling to recognize handwritten words without explicit character segmentation.

The model is designed to handle distortions and variations in real-world handwriting.

---

## Key Features
- End-to-end handwritten word recognition
- CNN feature extraction with ResNet
- BiLSTM based sequence modeling
- CTC Loss for alignment-free decoding
- Distortion-aware training
- Robust inference pipeline

---

## Technologies Used
- Python
- PyTorch
- CRNN
- ResNet34
- BiLSTM
- CTC Loss
- OpenCV

---

## Model Architecture


Input Image
↓
CNN Feature Extractor (ResNet34)
↓
Feature Map → Sequence Representation
↓
BiLSTM Layers
↓
CTC Decoder
↓
Predicted Word


---

## Dataset
The training dataset consists of handwritten word images.

Typical structure:


data/
├── train/
│ ├── images
│ └── labels
│
├── test/
│ ├── images
│ └── labels


Dataset sources may include:
- IAM Handwriting Dataset
- Kaggle Handwritten Word datasets

---

## Project Structure


handwritten-ocr-crnn/
│
├── model/
│ └── crnn_model.py
│
├── dataset/
│ └── loader.py
│
├── train.py
├── inference.py
│
├── utils/
│ └── preprocessing.py
│
├── requirements.txt
└── README.md

---
## Installation

```bash
git clone https://github.com/yourusername/handwritten-ocr-crnn.git
cd handwritten-ocr-crnn
pip install -r requirements.txt
Training
python train.py
Run Inference
python inference.py --image sample.png
Example Output

Input Image:
handwritten word image

Predicted Text:
hello



Future Improvements:
Transformer based OCR
Language model integration
Real-time OCR API deployment
