# CardioDiagnose

12-Lead ECG Arrhythmia Classification with Deep Learning

---

## Model Architecture
                      ECG SIGNAL (5000 x 12)
                              │
                              ↓
                    ┌─────────────────┐
                    │   Conv1D (64)   │
                    │   Kernel: 10    │
                    │   Activation: ReLU
                    └────────┬────────┘
                              │
                              ↓
                    ┌─────────────────┐
                    │   MaxPooling1D  │
                    │     Pool: 5     │
                    └────────┬────────┘
                              │
                              ↓
                    ┌─────────────────┐
                    │   Conv1D (128)  │
                    │   Kernel: 10    │
                    │   Activation: ReLU
                    └────────┬────────┘
                              │
                              ↓
                    ┌─────────────────┐
                    │   MaxPooling1D  │
                    │     Pool: 5     │
                    └────────┬────────┘
                              │
                              ↓
                    ┌─────────────────┐
                    │   Conv1D (256)  │
                    │   Kernel: 10    │
                    │   Activation: ReLU
                    └────────┬────────┘
                              │
                              ↓
                    ┌─────────────────┐
                    │ GlobalAveragePool│
                    └────────┬────────┘
                              │
                              ↓
                    ┌─────────────────┐
                    │    Dense (128)  │
                    │   Dropout (0.3) │
                    └────────┬────────┘
                              │
                              ↓
                    ┌─────────────────┐
                    │    Dense (64)   │
                    │   Dropout (0.3) │
                    └────────┬────────┘
                              │
                              ↓
                    ┌─────────────────┐
                    │    Dense (5)    │
                    │   Softmax       │
                    └────────┬────────┘
                              │
                              ↓
                    ┌─────────────────┐
                    │   Normal  AFib  │
                    │   PVC     Tach  │
                    │   Brady         │
                    └─────────────────┘

---
## ECG Waveforms

**Normal**     ▁▂▃▂▁    ▁▂▃▂▁    ▁▂▃▂▁    ▁▂▃▂▁
**AFib**       ▁▂▁▃▁▂▁▃▁▂▁▃▁▂▁▃▁▂▁▃▁▂▁▃▁▂
**PVC**        ▁▂▃▂▁    ▁▂▃▂▁    ▁▂▃▂▃▂▁    ▁▂▃▂▁
**Tachy**      ▁▂▃▂▁▂▃▂▁▂▃▂▁▂▃▂▁▂▃▂▁▂▃▂▁▂▃▂
**Brady**      ▁▂▃▂▁        ▁▂▃▂▁        ▁▂▃▂▁

## Detectable Arrhythmias

The system currently detects 5 types of cardiac rhythms:

Normal Sinus Rhythm
- Rate: 60-100 bpm
- Rhythm: Regular
- P waves: Present and normal
- PR interval: 120-200 ms

Atrial Fibrillation (AFib)
- Rate: Variable, usually fast
- Rhythm: Irregularly irregular
- P waves: Absent
- Clinical: 5x increased stroke risk

Premature Ventricular Contraction (PVC)
- Rhythm: Regular with premature beats
- QRS: Wide (>120 ms), bizarre morphology
- Compensatory pause present
- Clinical: Can indicate heart disease

Sinus Tachycardia
- Rate: >100 bpm
- Rhythm: Regular
- P waves: Present
- Clinical: May indicate fever, anxiety, heart failure

Sinus Bradycardia
- Rate: <60 bpm
- Rhythm: Regular
- P waves: Present
- Clinical: Can be normal in athletes, or indicate heart block

---

# Training Progress:

Epoch 1/50: █░░░░░░░░░░░░░░░░░░░ 5%   loss: 1.82 - acc: 0.42
Epoch 5/50: ███░░░░░░░░░░░░░░░░░ 15%  loss: 1.21 - acc: 0.58
Epoch 10/50:████░░░░░░░░░░░░░░░░ 25%  loss: 0.92 - acc: 0.67
Epoch 15/50:██████░░░░░░░░░░░░░░ 35%  loss: 0.74 - acc: 0.74
Epoch 20/50:████████░░░░░░░░░░░░ 45%  loss: 0.61 - acc: 0.79
Epoch 25/50:██████████░░░░░░░░░░ 55%  loss: 0.52 - acc: 0.83
Epoch 30/50:████████████░░░░░░░░ 65%  loss: 0.44 - acc: 0.86
Epoch 35/50:██████████████░░░░░░ 75%  loss: 0.38 - acc: 0.88
Epoch 40/50:████████████████░░░░ 85%  loss: 0.33 - acc: 0.90
Epoch 45/50:██████████████████░░ 95%  loss: 0.29 - acc: 0.91
Epoch 50/50:████████████████████ 100% loss: 0.26 - acc: 0.92
---

# Inference Speed:

Platform          Time per ECG    RAM Usage
─────────────────────────────────────────────
CPU (i7)          0.3 seconds     180 MB
CPU (i5)          0.5 seconds     180 MB
GPU (GTX 1060)    0.08 seconds    220 MB
GPU (A100)        0.02 seconds    240 MB
Raspberry Pi 4    1.8 seconds     120 MB
---

# API Endpoints:

GET  /health
     └── Response: { "status": "healthy", "model": "loaded" }

POST /predict
     ├── Body: multipart/form-data
     │   └── ecg_file: [ECG .dat file]
     └── Response: {
           "success": true,
           "diagnosis": "Atrial Fibrillation",
           "confidence": 0.94,
           "class_id": 1,
           "processing_time": 0.32
       }

GET  /info
     └── Response: {
           "classes": ["Normal", "AFib", "PVC", "Tachycardia", "Bradycardia"],
           "input_shape": [5000, 12],
           "sampling_rate": 500,
           "accuracy": 0.912
       }
---

🔬 Model Interpretability (Grad-CAM):

Original ECG:     ╭─╮   ╭─╮   ╭─╮   ╭─╮   ╭─╮
                 ╭╯ ╰╮ ╭╯ ╰╮ ╭╯ ╰╮ ╭╯ ╰╮ ╭╯ ╰╮
                ╭╯   ╰╮╯   ╰╮╯   ╰╮╯   ╰╮╯   ╰╮
                │     │     │     │     │     │

Grad-CAM Heatmap:
                 ░░░▓▓███▓▓░░░▓▓███▓▓░░░▓▓███▓▓
                 ░▓███████▓░▓███████▓░▓███████▓
                 ███████████████████████████████
                 ░▓███████▓░▓███████▓░▓███████▓
                 ░░░▓▓███▓▓░░░▓▓███▓▓░░░▓▓███▓▓

Focus Areas:         ↑         ↑         ↑
                QRS Complex  P Wave   T Wave
---

# Comparison with Other Methods:

Method              Accuracy  Sensitivity  Specificity  Paper
────────────────────────────────────────────────────────────────
CardioDiagnose (CNN)  91.2%     90.5%       91.8%      This work
Hannun et al. (2019)  90.8%     90.2%       91.3%      Nature Medicine
Rajpurkar et al.     91.5%     91.0%       92.1%      arXiv 2017
Ribeiro et al.       90.1%     89.7%       90.5%      Nature Comms 2020
---

# Clinical Validation Status:

┌─────────────────────────────────────────────────┐
│ Phase 1: Algorithm development    ✅ Complete   │
│ Phase 2: Internal validation      ✅ Complete   │
│ Phase 3: External validation      ⏳ In progress│
│ Phase 4: Prospective study        ⏳ Planned    │
│ Phase 5: Regulatory approval      ⏳ Planned    │
└─────────────────────────────────────────────────┘
---

# Code Example:

`python
# Load a patient ECG
record = wfdb.rdrecord('patient_100')

# Preprocess
signal = preprocess_signal(record.p_signal)

# Predict
model = tf.keras.models.load_model('best_model.h5')
pred = model.predict(signal.reshape(1, 5000, 12))
class_idx = np.argmax(pred[0])

# Result
print(f"Diagnosis: {CLASS_NAMES[class_idx]}")
print(f"Confidence: {pred[0][class_idx]:.2%}")
`

---

