from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import wfdb
import os
from config import Config

app = Flask(name)
app.config.from_object(Config)

# Load model
model = tf.keras.models.load_model('best_model.h5')

# Class names
CLASS_NAMES = ['Normal', 'Atrial Fibrillation', 'PVC', 'Tachycardia', 'Bradycardia']

@app.route('/')
def index():
    return render_template('index.html')  # بعداً می‌سازیم

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get ECG file
        file = request.files['ecg_file']
        file_path = 'temp_ecg.dat'
        file.save(file_path)
        
        # Load ECG record
        record = wfdb.rdrecord(file_path.replace('.dat', ''))
        signal = record.p_signal
        
        # Preprocess
        signal = (signal - np.mean(signal)) / np.std(signal)
        if len(signal) > 5000:
            signal = signal[:5000]
        elif len(signal) < 5000:
            pad = 5000 - len(signal)
            signal = np.pad(signal, ((0, pad), (0, 0)), mode='constant')
        
        # Predict
        signal = signal.reshape(1, 5000, 12)
        prediction = model.predict(signal)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        
        # Clean up
        os.remove(file_path)
        for ext in ['.dat', '.hea']:
            if os.path.exists(file_path.replace('.dat', ext)):
                os.remove(file_path.replace('.dat', ext))
        
        return jsonify({
            'success': True,
            'diagnosis': CLASS_NAMES[class_idx],
            'confidence': confidence,
            'class_id': int(class_idx)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if name == 'main':
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
