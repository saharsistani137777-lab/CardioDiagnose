import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from preprocess import load_all_records, preprocess_signal
from model import create_ecg_cnn
from config import Config

def prepare_data():
    """Load and prepare ECG data"""
    print("Loading ECG records...")
    X, y = load_all_records(Config.DATA_PATH)
    
    print(f"Loaded {len(X)} records")
    print(f"Signal shape: {X[0].shape}")
    
    # Preprocess signals
    X_processed = []
    for signal in X:
        processed = preprocess_signal(signal)
        X_processed.append(processed)
    
    X = np.array(X_processed)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=Config.VALIDATION_SPLIT + Config.TEST_SPLIT,
        random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=Config.TEST_SPLIT/(Config.VALIDATION_SPLIT + Config.TEST_SPLIT),
        random_state=42, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model():
    """Train the ECG classification model"""
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    
    # Create model
    model = create_ecg_cnn(
        input_shape=(Config.SAMPLE_RATE * Config.DURATION, Config.NUM_LEADS),
        num_classes=Config.NUM_CLASSES
    )
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint('best_model.h5', save_best_only=True),
        EarlyStopping(patience=10, restore_best_weights=True)
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.show()
    
    return model, history

if name == "main":
    model, history = train_model()
