import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve, 
                            confusion_matrix, classification_report)
from lstm_model import LSTMMotorModel, create_synthetic_training_data
from bert_model import SimpleTextCoherenceModel, create_synthetic_text_data
from ensemble import EnsembleScorer
from sklearn.model_selection import train_test_split
import os

class ModelEvaluator:
    def __init__(self, save_plots: bool = True):
        self.save_plots = save_plots
        self.results = {}
        
    def evaluate_motor_model(self, model: LSTMMotorModel, X_test: np.ndarray, y_test: np.ndarray):
        print("\n" + "="*60)
        print("Evaluating LSTM Motor Model")
        print("="*60)
        
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_prob)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Concern']))
        
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        self.results['motor'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob
        }
        
        if self.save_plots:
            self._plot_roc_curve(y_test, y_pred_prob, 'Motor Model')
            self._plot_confusion_matrix(cm, 'Motor Model')
        
        return self.results['motor']
    
    def evaluate_text_model(self, model: SimpleTextCoherenceModel, texts_test: list, y_test: np.ndarray):
        print("\n" + "="*60)
        print("Evaluating Text Coherence Model")
        print("="*60)
        
        y_pred_prob = model.predict(texts_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_prob)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Coherent', 'Incoherent']))
        
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        self.results['text'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob
        }
        
        if self.save_plots:
            self._plot_roc_curve(y_test, y_pred_prob, 'Text Model')
            self._plot_confusion_matrix(cm, 'Text Model')
        
        return self.results['text']
    
    def evaluate_ensemble(self, motor_scores: np.ndarray, text_scores: np.ndarray, 
                         y_test: np.ndarray, ensemble: EnsembleScorer):
        print("\n" + "="*60)
        print("Evaluating Ensemble Model")
        print("="*60)
        
        ensemble_scores = ensemble.batch_combine_scores(motor_scores, text_scores)
        y_pred = (ensemble_scores > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, ensemble_scores)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Concern']))
        
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        self.results['ensemble'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_prob': ensemble_scores
        }
        
        if self.save_plots:
            self._plot_roc_curve(y_test, ensemble_scores, 'Ensemble Model')
            self._plot_confusion_matrix(cm, 'Ensemble Model')
            self._plot_score_comparison(motor_scores, text_scores, ensemble_scores, y_test)
        
        return self.results['ensemble']
    
    def _plot_roc_curve(self, y_true, y_pred_prob, model_name):
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        auc = roc_auc_score(y_true, y_pred_prob)
        
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        
        filename = f"saved_models/roc_curve_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to {filename}")
    
    def _plot_confusion_matrix(self, cm, model_name):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Normal', 'Concern'],
                   yticklabels=['Normal', 'Concern'])
        
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        
        filename = f"saved_models/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {filename}")
    
    def _plot_score_comparison(self, motor_scores, text_scores, ensemble_scores, y_true):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        colors = ['green' if y == 0 else 'red' for y in y_true]
        
        axes[0].scatter(range(len(motor_scores)), motor_scores, c=colors, alpha=0.6)
        axes[0].axhline(y=0.5, color='black', linestyle='--', linewidth=1)
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Motor Score')
        axes[0].set_title('Motor Scores')
        axes[0].grid(alpha=0.3)
        
        axes[1].scatter(range(len(text_scores)), text_scores, c=colors, alpha=0.6)
        axes[1].axhline(y=0.5, color='black', linestyle='--', linewidth=1)
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Text Score')
        axes[1].set_title('Text Scores')
        axes[1].grid(alpha=0.3)
        
        axes[2].scatter(range(len(ensemble_scores)), ensemble_scores, c=colors, alpha=0.6)
        axes[2].axhline(y=0.5, color='black', linestyle='--', linewidth=1)
        axes[2].set_xlabel('Sample Index')
        axes[2].set_ylabel('Ensemble Score')
        axes[2].set_title('Ensemble Scores')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        filename = "saved_models/score_comparison.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Score comparison plot saved to {filename}")

def main():
    print("\n" + "="*70)
    print(" "*15 + "MEMORY GUARDIAN - MODEL EVALUATION")
    print("="*70)
    
    print("\nGenerating test data...")
    X_motor, y_motor = create_synthetic_training_data(200, sequence_length=10, n_features=5)
    texts, y_text = create_synthetic_text_data(200)
    
    print("\nLoading models...")
    motor_model = LSTMMotorModel(sequence_length=10, n_features=5)
    motor_model.load_model()
    
    text_model = SimpleTextCoherenceModel(vocab_size=2000, max_length=50)
    text_model.load_model()
    
    evaluator = ModelEvaluator(save_plots=True)
    
    motor_results = evaluator.evaluate_motor_model(motor_model, X_motor, y_motor)
    
    text_results = evaluator.evaluate_text_model(text_model, texts, y_text)
    
    motor_scores = motor_model.predict(X_motor)
    text_scores = text_model.predict(texts)
    
    y_combined = np.logical_or(y_motor, y_text).astype(int)
    
    ensemble = EnsembleScorer(motor_weight=0.6, language_weight=0.4)
    ensemble_results = evaluator.evaluate_ensemble(motor_scores, text_scores, y_combined, ensemble)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nMotor Model   - Accuracy: {motor_results['accuracy']:.4f}, AUC: {motor_results['auc']:.4f}")
    print(f"Text Model    - Accuracy: {text_results['accuracy']:.4f}, AUC: {text_results['auc']:.4f}")
    print(f"Ensemble      - Accuracy: {ensemble_results['accuracy']:.4f}, AUC: {ensemble_results['auc']:.4f}")
    print(f"\nPlots saved to: saved_models/")

if __name__ == "__main__":
    main()
