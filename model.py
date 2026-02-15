import tensorflow as tf
from tensorflow.keras import layers, models
from config import Config

def create_ecg_cnn(input_shape=(5000, 12), num_classes=5):
    """Create 1D CNN for ECG classification"""
    model = models.Sequential([
        # First convolutional block
        layers.Conv1D(64, 10, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(5),
        layers.Dropout(0.2),
        
        # Second convolutional block
        layers.Conv1D(128, 10, activation='relu'),
        layers.MaxPooling1D(5),
        layers.Dropout(0.2),
        
        # Third convolutional block
        layers.Conv1D(256, 10, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_lstm_model(input_shape=(5000, 12), num_classes=5):
    """Create LSTM model for ECG classification"""
    model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_hybrid_model(input_shape=(5000, 12), num_classes=5):
    """CNN + LSTM hybrid model"""
    inputs = layers.Input(shape=input_shape)
    
    # CNN branch
    cnn = layers.Conv1D(64, 10, activation='relu')(inputs)
    cnn = layers.MaxPooling1D(5)(cnn)
    cnn = layers.Conv1D(128, 10, activation='relu')(cnn)
    cnn = layers.GlobalAveragePooling1D()(cnn)
    
    # LSTM branch
    lstm = layers.LSTM(64, return_sequences=True)(inputs)
    lstm = layers.LSTM(32)(lstm)
    
    # Concatenate
    concat = layers.Concatenate()([cnn, lstm])
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(concat)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
