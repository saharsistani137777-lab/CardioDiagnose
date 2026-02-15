import wfdb
import numpy as np
import pandas as pd
import os
from config import Config

def load_ecg_data(record_path):
    """Load a single ECG record"""
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal
    return signal

def load_all_records(data_path):
    """Load all ECG records from database"""
    # Find all .dat files
    records = []
    labels = []
    
    # Load metadata
    meta = pd.read_csv(os.path.join(data_path, 'ptbxl_database.csv'))
    
    for idx, row in meta.iterrows():
        record_path = os.path.join(data_path, row['filename_hr'])
        signal = load_ecg_data(record_path.replace('.dat', ''))
        records.append(signal)
        
        # Extract arrhythmia label
        label = extract_label(row)
        labels.append(label)
    
    return np.array(records), np.array(labels)

def extract_label(row):
    """Extract arrhythmia type from metadata"""
    scp_codes = eval(row['scp_codes'])
    
    # Define arrhythmia classes
    if 'AFIB' in scp_codes:
        return 1  # Atrial Fibrillation
    elif 'PVC' in scp_codes:
        return 2  # Premature Ventricular Contraction
    elif 'STACH' in scp_codes:
        return 3  # Sinus Tachycardia
    elif 'SBRAD' in scp_codes:
        return 4  # Sinus Bradycardia
    else:
        return 0  # Normal

def preprocess_signal(signal, target_length=5000):
    """Normalize and resize ECG signal"""
    # Normalize
    signal = (signal - np.mean(signal)) / np.std(signal)
    
    # Resize if needed
    if len(signal) > target_length:
        signal = signal[:target_length]
    elif len(signal) < target_length:
        pad = target_length - len(signal)
        signal = np.pad(signal, ((0, pad), (0, 0)), mode='constant')
    
    return signal
