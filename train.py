import numpy as np
from sklearn.model_selection import train_test_split
from lstm_model import LSTMMotorModel, create_synthetic_training_data
from bert_model import SimpleTextCoherenceModel, create_synthetic_text_data
from utils import ensure_directories, save_config
import json

def train_motor_model(n_samples: int = 1000, sequence_length: int = 10, 
                     n_features: int = 5, epochs: int = 50):
    print("\n" + "="*60)
    print("Training LSTM Motor Model")
    print("="*60)
    
    print(f"\nGenerating {n_samples} synthetic training samples...")
    X, y = create_synthetic_training_data(n_samples, sequence_length, n_features)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Positive samples: {np.sum(y_train)}/{len(y_train)} ({np.mean(y_train)*100:.1f}%)")
    
    model = LSTMMotorModel(sequence_length=sequence_length, n_features=n_features)
    model.build_model()
    
    print("\nModel Architecture:")
    model.model.summary()
    
    print(f"\nTraining for {epochs} epochs...")
    history = model.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=32)
    
    print("\nEvaluating on test set...")
    test_results = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test AUC: {test_results['auc']:.4f}")
    
    print("\nSaving model...")
    model.save_model()
    
    config = {
        'model_type': 'LSTM Motor Model',
        'sequence_length': sequence_length,
        'n_features': n_features,
        'training_samples': n_samples,
        'test_accuracy': float(test_results['accuracy']),
        'test_auc': float(test_results['auc'])
    }
    
    return model, history, test_results, config

def train_text_model(n_samples: int = 600, vocab_size: int = 2000, 
                    max_length: int = 50, epochs: int = 30):
    print("\n" + "="*60)
    print("Training Text Coherence Model")
    print("="*60)
    
    print(f"\nGenerating {n_samples} synthetic text samples...")
    texts, labels = create_synthetic_text_data(n_samples)
    
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    texts_train, texts_val, y_train, y_val = train_test_split(
        texts_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {len(texts_train)} samples")
    print(f"Validation set: {len(texts_val)} samples")
    print(f"Test set: {len(texts_test)} samples")
    print(f"Positive samples: {np.sum(y_train)}/{len(y_train)} ({np.mean(y_train)*100:.1f}%)")
    
    model = SimpleTextCoherenceModel(vocab_size=vocab_size, max_length=max_length)
    model.build_model()
    
    print("\nModel Architecture:")
    model.model.summary()
    
    print(f"\nTraining for {epochs} epochs...")
    history = model.train(texts_train, y_train, texts_val, y_val, epochs=epochs, batch_size=32)
    
    print("\nEvaluating on test set...")
    test_results = model.evaluate(texts_test, y_test)
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test AUC: {test_results['auc']:.4f}")
    
    print("\nSaving model...")
    model.save_model()
    
    config = {
        'model_type': 'Text Coherence Model',
        'vocab_size': vocab_size,
        'max_length': max_length,
        'training_samples': n_samples,
        'test_accuracy': float(test_results['accuracy']),
        'test_auc': float(test_results['auc'])
    }
    
    return model, history, test_results, config

def main():
    ensure_directories()
    
    print("\n" + "="*70)
    print(" "*15 + "MEMORY GUARDIAN - MODEL TRAINING")
    print("="*70)
    
    motor_model, motor_history, motor_results, motor_config = train_motor_model(
        n_samples=1000,
        sequence_length=10,
        n_features=5,
        epochs=50
    )
    
    text_model, text_history, text_results, text_config = train_text_model(
        n_samples=600,
        vocab_size=2000,
        max_length=50,
        epochs=30
    )
    
    combined_config = {
        'motor_model': motor_config,
        'text_model': text_config,
        'ensemble_weights': {
            'motor': 0.6,
            'language': 0.4
        }
    }
    
    save_config(combined_config)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nMotor Model - Accuracy: {motor_results['accuracy']:.4f}, AUC: {motor_results['auc']:.4f}")
    print(f"Text Model - Accuracy: {text_results['accuracy']:.4f}, AUC: {text_results['auc']:.4f}")
    print(f"\nModels saved to: saved_models/")
    print(f"Configuration saved to: saved_models/config.json")
    
    return motor_model, text_model, combined_config

if __name__ == "__main__":
    main()
