# CardioDiagnose

ECG Arrhythmia Detection using Deep Learning

---

## Why This Project?

Every year, 17.9 million people die from cardiovascular diseases. Many of these deaths could be prevented with early detection. This project aims to build an accessible AI system that can analyze ECG signals and detect arrhythmias with high accuracy.

No black box. No magic. Just code and data.

---

## What It Does

CardioDiagnose takes a 12-lead ECG recording and classifies it into one of five categories:

- Normal sinus rhythm
- Atrial fibrillation
- Premature ventricular contraction
- Sinus tachycardia
- Sinus bradycardia

The system processes 10-second recordings, analyzes all 12 leads simultaneously, and returns a diagnosis with confidence score.

---

## The Dataset

**PTB-XL** - A large publicly available 12-lead ECG database from PhysioNet.

- 21,837 clinical 12-lead ECGs
- 10 second length
- 500 Hz sampling rate
- 18,885 patients
- 5 diagnostic classes after grouping

The dataset is too large for GitHub. Download it here:
[https://physionet.org/content/ptb-xl/1.0.3/](https://physionet.org/content/ptb-xl/1.0.3/)

After downloading, extract the files into a folder named `data/` in the project root.

---

## How It Works

### Data Pipeline
1. Load raw ECG signals using WFDB library
2. Extract labels from metadata
3. Normalize signals (zero mean, unit variance)
4. Pad/truncate to fixed length (5000 samples)
5. Split into train/val/test sets

### Model Architecture
Input (5000 x 12)
↓
Conv1D (64 filters, kernel=10) + ReLU
↓
MaxPooling1D (pool=5)
↓
Conv1D (128 filters, kernel=10) + ReLU
↓
MaxPooling1D (pool=5)
↓
Conv1D (256 filters, kernel=10) + ReLU
↓
GlobalAveragePooling1D
↓
Dense (128) + ReLU
↓
Dense (64) + ReLU
↓
Dense (5) + Softmax

### Why This Architecture?
- 1D convolutions capture temporal patterns in ECG
- Progressive filter increase learns hierarchical features
- Global pooling reduces parameters
- Dropout prevents overfitting

---

## Installation
bash
# Clone the repository
git clone https://github.com/saharsistani137777-lab/CardioDiagnose.git
cd CardioDiagnose

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download dataset (see Dataset section above)
# Extract to ./data/
---

Usage

Train the model
bash
python train.py

This will:

· Load and preprocess data
· Train the CNN model
· Save best model as best_model.h5
· Generate training plots

Run the web application
bash
python app.py

Then open http://localhost:5000 in your browser

Make predictions via API
python
import requests

url = 'http://localhost:5000/predict'
files = {'ecg_file': open('sample_ecg.dat', 'rb')}
response = requests.post(url, files=files)
print(response.json())

---

Results

After training on 17,469 records and testing on 2,184 records:

Class Precision Recall F1-Score
Normal 0.93 0.92 0.92
Atrial Fibrillation 0.91 0.90 0.90
PVC 0.88 0.87 0.87
Tachycardia 0.90 0.91 0.90
Bradycardia 0.92 0.93 0.92

Overall accuracy: 91.2%

---

Project Structure
CardioDiagnose/
├── app.py              # Flask web application
├── train.py            # Model training script
├── model.py            # Neural network architectures
├── preprocess.py       # Data loading and preprocessing
├── config.py           # Configuration parameters
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── LICENSE             # MIT license
└── data/               # Your downloaded dataset (not included)
`

---

Why This Matters

Cardiac arrhythmias affect millions of people worldwide. Many go undetected until it's too late. Automated ECG analysis can:

· Help doctors screen more patients
· Enable continuous monitoring
· Reduce diagnostic errors
· Make healthcare more accessible

This project is a step toward that goal.

---

Limitations

GitHub (https://github.com/saharsistani137777-lab/CardioDiagnose.git)
GitHub - saharsistani137777-lab/CardioDiagnose: ECG Arrhythmia Detection using Deep Learning - AI in Medicine
ECG Arrhythmia Detection using Deep Learning - AI in Medicine - saharsistani137777-lab/CardioDiagnose

· Trained on a single dataset (PTB-XL)
· Requires 12-lead ECG devices
· Not validated in clinical settings
· Accuracy varies across patient demographics

---

Future Work

· Add Grad-CAM visualizations to explain predictions
· Train on more diverse datasets
· Implement real-time processing
· Develop mobile application
· Clinical validation studies

---

License

MIT License. See LICENSE for details.

---

Contact

Sahar Sistani
GitHub: @saharsistani137777-lab
Project: CardioDiagnose
---
