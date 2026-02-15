import os

class Config:
    # Data
    DATA_PATH = 'data/ptb-xl'
    SAMPLE_RATE = 500
    DURATION = 10
    NUM_LEADS = 12
    NUM_CLASSES = 5
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Flask
    SECRET_KEY = 'dev-key-2026'
    DEBUG = True
    PORT = 5000
    HOST = '0.0.0.0'
