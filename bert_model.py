import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import List, Optional
from utils import get_model_path
import os
import re

class SimpleTextCoherenceModel:
    def __init__(self, vocab_size: int = 10000, max_length: int = 100, embedding_dim: int = 128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.model = None
        self.tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        self.model_name = 'text_coherence_model'
        
    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def fit_tokenizer(self, texts: List[str]):
        preprocessed = [self.preprocess_text(t) for t in texts]
        self.tokenizer.fit_on_texts(preprocessed)
    
    def texts_to_sequences(self, texts: List[str]) -> np.ndarray:
        preprocessed = [self.preprocess_text(t) for t in texts]
        sequences = self.tokenizer.texts_to_sequences(preprocessed)
        padded = keras.preprocessing.sequence.pad_sequences(
            sequences, 
            maxlen=self.max_length, 
            padding='post',
            truncating='post'
        )
        return padded
    
    def build_model(self) -> keras.Model:
        model = models.Sequential([
            layers.Input(shape=(self.max_length,)),
            
            layers.Embedding(self.vocab_size, self.embedding_dim, mask_zero=True),
            
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.3),
            
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        self.model = model
        return model
    
    def train(self, texts: List[str], y_train: np.ndarray,
              val_texts: Optional[List[str]] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 30, batch_size: int = 32) -> keras.callbacks.History:
        if self.model is None:
            self.build_model()
        
        self.fit_tokenizer(texts)
        X_train = self.texts_to_sequences(texts)
        
        validation_data = None
        if val_texts is not None and y_val is not None:
            X_val = self.texts_to_sequences(val_texts)
            validation_data = (X_val, y_val)
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        X = self.texts_to_sequences(texts)
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def predict_single(self, text: str) -> float:
        prediction = self.predict([text])
        return float(prediction[0])
    
    def save_model(self, filepath: Optional[str] = None):
        if self.model is None:
            raise ValueError("No model to save")
        
        if filepath is None:
            filepath = get_model_path(self.model_name)
        
        self.model.save(filepath)
        
        tokenizer_path = filepath.replace('.keras', '_tokenizer.json')
        tokenizer_json = self.tokenizer.to_json()
        with open(tokenizer_path, 'w') as f:
            f.write(tokenizer_json)
        
        print(f"Model saved to {filepath}")
        print(f"Tokenizer saved to {tokenizer_path}")
    
    def load_model(self, filepath: Optional[str] = None):
        if filepath is None:
            filepath = get_model_path(self.model_name)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = keras.models.load_model(filepath)
        
        tokenizer_path = filepath.replace('.keras', '_tokenizer.json')
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'r') as f:
                tokenizer_json = f.read()
            self.tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
            print(f"Tokenizer loaded from {tokenizer_path}")
        else:
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}. Model cannot be used without tokenizer.")
        
        print(f"Model loaded from {filepath}")
    
    def evaluate(self, texts: List[str], y_test: np.ndarray) -> dict:
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        X_test = self.texts_to_sequences(texts)
        loss, accuracy, auc = self.model.evaluate(X_test, y_test, verbose=0)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'auc': auc
        }

def create_synthetic_text_data(n_samples: int = 500) -> tuple:
    coherent_samples = [
        "The quick brown fox jumps over the lazy dog.",
        "I enjoy reading books about science and technology.",
        "The weather today is beautiful and sunny.",
        "She walks to the park every morning for exercise.",
        "Technology has changed how we communicate with each other.",
        "The cat sat on the mat and fell asleep.",
        "Learning new skills takes time and practice.",
        "The restaurant serves delicious food and has great service.",
        "He plays guitar in a band on weekends.",
        "The mountains are covered with snow in winter."
    ]
    
    incoherent_samples = [
        "Tree sky walking tomorrow umbrella table.",
        "Computer happy running never purple elephant.",
        "Sandwich music flying yesterday green Monday.",
        "Window jumping sleep orange telephone happy.",
        "Birthday running computer never sky elephant.",
        "Coffee flying yesterday purple telephone window.",
        "Guitar happy Monday green umbrella table.",
        "Banana running computer never sky elephant.",
        "Telephone flying yesterday purple happy window.",
        "Monday happy running computer never elephant."
    ]
    
    texts = []
    labels = []
    
    for _ in range(n_samples // 2):
        texts.append(np.random.choice(coherent_samples))
        labels.append(0)
    
    for _ in range(n_samples // 2):
        texts.append(np.random.choice(incoherent_samples))
        labels.append(1)
    
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = np.array([labels[i] for i in indices])
    
    return texts, labels

if __name__ == "__main__":
    print("Creating synthetic text data...")
    texts_train, y_train = create_synthetic_text_data(400)
    texts_val, y_val = create_synthetic_text_data(100)
    
    print(f"Training samples: {len(texts_train)}, Validation samples: {len(texts_val)}")
    
    model = SimpleTextCoherenceModel(vocab_size=1000, max_length=50)
    model.build_model()
    
    print("\nModel architecture:")
    model.model.summary()
    
    print("\nTraining model...")
    history = model.train(texts_train, y_train, texts_val, y_val, epochs=10)
    
    print("\nEvaluating model...")
    results = model.evaluate(texts_val, y_val)
    print(f"Validation results: {results}")
    
    print("\nSaving model...")
    model.save_model()
