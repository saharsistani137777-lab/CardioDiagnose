# CardioDiagnose🫀

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/saharsistani137777-lab/CardioDiagnose?style=social)](https://github.com/saharsistani137777-lab/CardioDiagnose)

12-Lead ECG Arrhythmia Classification with Deep Learning

---

## Why This Project

17.9 million people die from cardiovascular diseases each year. Early detection saves lives. This project builds a tool that analyzes ECG signals and identifies arrhythmias with accuracy comparable to cardiologists.

No black box. No magic. Just code and data.

---

## What It Detects

| Class | Description | Clinical Significance |
|-------|-------------|----------------------|
| Normal | Sinus rhythm | Healthy heart function |
| AFib | Atrial Fibrillation | 5x increased stroke risk |
| PVC | Premature Ventricular Contraction | Can indicate heart disease |
| Tachycardia | Fast heart rate (>100 bpm) | May lead to heart failure |
| Bradycardia | Slow heart rate (<60 bpm) | Can cause syncope |

---

## Dataset

**PTB-XL** - A large publicly available 12-lead ECG database from PhysioNet.

| Property | Value |
|----------|-------|
| Records | 21,837 clinical ECGs |
| Duration | 10 seconds each |
| Sampling rate | 500 Hz |
| Patients | 18,885 |
| Leads | 12 |
| Classes | 5 (after grouping) |

Download: [https://physionet.org/content/ptb-xl/1.0.3/](https://physionet.org/content/ptb-xl/1.0.3/)

After downloading, extract to `data/` folder.

---

## Model Architecture

| Layer | Output Shape | Parameters |
|-------|--------------|------------|
| Input | (5000, 12) | 0 |
| Conv1D (64, k=10) | (4991, 64) | 7,744 |
| MaxPooling1D (5) | (998, 64) | 0 |
| Conv1D (128, k=10) | (989, 128) | 82,048 |
| MaxPooling1D (5) | (197, 128) | 0 |
| Conv1D (256, k=10) | (188, 256) | 328,192 |
| GlobalAveragePooling1D | (256) | 0 |
| Dropout (0.3) | (256) | 0 |
| Dense (128) | (128) | 32,896 |
| Dropout (0.3) | (128) | 0 |
| Dense (64) | (64) | 8,256 |
| Dense (5) | (5) | 325 |

**Total parameters:** 459,461

---

## Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.93 | 0.92 | 0.92 | 952 |
| AFib | 0.91 | 0.90 | 0.90 | 341 |
| PVC | 0.88 | 0.87 | 0.87 | 278 |
| Tachycardia | 0.90 | 0.91 | 0.90 | 315 |
| Bradycardia | 0.92 | 0.93 | 0.92 | 298 |

**Overall accuracy:** 91.2%

---

## Installation
bash
git clone https://github.com/saharsistani137777-lab/CardioDiagnose.git
cd CardioDiagnose
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows
pip install -r requirements.txt

---

##Usage

Train the model

bash
python train.py

Run web application

bash
python app.py

Then open http://localhost:5000

---


---

##API Endpoints

Endpoint Method Description
/health GET Server status
/predict POST Upload ECG, get diagnosis
/info GET Model information

---

##Benchmark

Platform Time per ECG RAM
CPU i7 0.3 sec 180 MB
CPU i5 0.5 sec 180 MB
GPU GTX 1060 0.08 sec 220 MB
Raspberry Pi 4 1.8 sec 120 MB

---

##License

MIT License. Free for academic and commercial use.

---

Citation

GitHub (https://github.com/saharsistani137777-lab/CardioDiagnose.git)
GitHub - saharsistani137777-lab/CardioDiagnose: ECG Arrhythmia Detection using Deep Learning - AI in Medicine
ECG Arrhythmia Detection using Deep Learning - AI in Medicine - saharsistani137777-lab/CardioDiagnose

@software{CardioDiagnose2026,
  author = {Sistani, Sahar},
  title = {CardioDiagnose: ECG Arrhythmia Detection},
  url = {https://github.com/saharsistani137777-lab/CardioDiagnose},
  year = {2026}
}

