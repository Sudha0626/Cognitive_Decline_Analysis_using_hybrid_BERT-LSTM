import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Optional
from utils import get_model_path, model_exists
import os

class LSTMMotorModel:
    def __init__(self, sequence_length: int = 10, n_features: int = 5):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.model_name = 'lstm_motor_model'
        
    def build_model(self) -> keras.Model:
        model = models.Sequential([
            layers.Input(shape=(self.sequence_length, self.n_features)),
            
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.3),
            
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 50, batch_size: int = 32) -> keras.callbacks.History:
        if self.model is None:
            self.build_model()
            
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not built or loaded. Train or load a model first.")
            
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def predict_single(self, sequence: np.ndarray) -> float:
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, self.sequence_length, self.n_features)
            
        prediction = self.predict(sequence)
        return float(prediction[0])
    
    def save_model(self, filepath: Optional[str] = None):
        if self.model is None:
            raise ValueError("No model to save")
            
        if filepath is None:
            filepath = get_model_path(self.model_name)
            
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Optional[str] = None):
        if filepath is None:
            filepath = get_model_path(self.model_name)
            
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        if self.model is None:
            raise ValueError("Model not built or loaded")
            
        loss, accuracy, auc = self.model.evaluate(X_test, y_test, verbose=0)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'auc': auc
        }

def create_synthetic_training_data(n_samples: int = 1000, sequence_length: int = 10, 
                                   n_features: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.randn(n_samples, sequence_length, n_features)
    
    feature_importance = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    scores = np.sum(X[:, -3:, :] * feature_importance, axis=(1, 2))
    
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    y = (scores > 0.5).astype(np.float32)
    
    return X, y

if __name__ == "__main__":
    print("Creating synthetic training data...")
    X_train, y_train = create_synthetic_training_data(800)
    X_val, y_val = create_synthetic_training_data(200)
    
    print(f"Training data shape: {X_train.shape}, Labels: {y_train.shape}")
    
    model = LSTMMotorModel(sequence_length=10, n_features=5)
    model.build_model()
    
    print("\nModel architecture:")
    model.model.summary()
    
    print("\nTraining model...")
    history = model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
    
    print("\nEvaluating model...")
    results = model.evaluate(X_val, y_val)
    print(f"Validation results: {results}")
    
    print("\nSaving model...")
    model.save_model()
