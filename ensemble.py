import numpy as np
from typing import Tuple, Optional
from utils import normalize_score

class EnsembleScorer:
    def __init__(self, motor_weight: float = 0.6, language_weight: float = 0.4):
        if abs(motor_weight + language_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        
        self.motor_weight = motor_weight
        self.language_weight = language_weight
        
    def combine_scores(self, motor_score: float, language_score: float) -> float:
        motor_score = normalize_score(motor_score, 0.0, 1.0)
        language_score = normalize_score(language_score, 0.0, 1.0)
        
        final_score = (self.motor_weight * motor_score + 
                      self.language_weight * language_score)
        
        return normalize_score(final_score, 0.0, 1.0)
    
    def interpret_score(self, score: float) -> dict:
        score = normalize_score(score, 0.0, 1.0)
        
        if score < 0.3:
            risk_level = "Low"
            status = "Normal"
            color = "green"
            recommendation = "Cognitive patterns appear normal. Continue regular monitoring."
        elif score < 0.5:
            risk_level = "Low-Medium"
            status = "Slight Deviation"
            color = "lightgreen"
            recommendation = "Minor variations detected. Continue monitoring for trends."
        elif score < 0.7:
            risk_level = "Medium"
            status = "Moderate Concern"
            color = "orange"
            recommendation = "Noticeable changes detected. Consider consulting a healthcare professional."
        else:
            risk_level = "High"
            status = "Significant Concern"
            color = "red"
            recommendation = "Significant cognitive changes detected. Recommend medical evaluation."
        
        return {
            'score': round(score, 3),
            'risk_level': risk_level,
            'status': status,
            'color': color,
            'recommendation': recommendation,
            'percentage': round(score * 100, 1)
        }
    
    def batch_combine_scores(self, motor_scores: np.ndarray, 
                            language_scores: np.ndarray) -> np.ndarray:
        if len(motor_scores) != len(language_scores):
            raise ValueError("Motor and language score arrays must have same length")
        
        motor_scores = np.clip(motor_scores, 0.0, 1.0)
        language_scores = np.clip(language_scores, 0.0, 1.0)
        
        final_scores = (self.motor_weight * motor_scores + 
                       self.language_weight * language_scores)
        
        return np.clip(final_scores, 0.0, 1.0)
    
    def get_component_contributions(self, motor_score: float, 
                                   language_score: float) -> dict:
        motor_score = normalize_score(motor_score, 0.0, 1.0)
        language_score = normalize_score(language_score, 0.0, 1.0)
        
        final_score = self.combine_scores(motor_score, language_score)
        
        motor_contribution = self.motor_weight * motor_score
        language_contribution = self.language_weight * language_score
        
        return {
            'final_score': round(final_score, 3),
            'motor_score': round(motor_score, 3),
            'language_score': round(language_score, 3),
            'motor_contribution': round(motor_contribution, 3),
            'language_contribution': round(language_contribution, 3),
            'motor_percentage': round(motor_contribution / final_score * 100, 1) if final_score > 0 else 0,
            'language_percentage': round(language_contribution / final_score * 100, 1) if final_score > 0 else 0
        }

def test_ensemble():
    ensemble = EnsembleScorer(motor_weight=0.6, language_weight=0.4)
    
    test_cases = [
        (0.2, 0.1, "Low risk scenario"),
        (0.5, 0.4, "Medium risk scenario"),
        (0.8, 0.7, "High risk scenario"),
        (0.3, 0.8, "Language-dominant concern"),
        (0.9, 0.2, "Motor-dominant concern")
    ]
    
    print("Testing Ensemble Scorer\n" + "="*60)
    
    for motor, language, description in test_cases:
        result = ensemble.combine_scores(motor, language)
        interpretation = ensemble.interpret_score(result)
        contributions = ensemble.get_component_contributions(motor, language)
        
        print(f"\n{description}")
        print(f"Motor Score: {motor:.2f}, Language Score: {language:.2f}")
        print(f"Final Score: {result:.3f} ({interpretation['percentage']}%)")
        print(f"Risk Level: {interpretation['risk_level']} - {interpretation['status']}")
        print(f"Recommendation: {interpretation['recommendation']}")
        print(f"Contributions - Motor: {contributions['motor_contribution']:.3f} ({contributions['motor_percentage']:.1f}%), "
              f"Language: {contributions['language_contribution']:.3f} ({contributions['language_percentage']:.1f}%)")
    
    print("\n" + "="*60)
    print("\nBatch processing test:")
    motor_scores = np.array([0.2, 0.5, 0.8])
    language_scores = np.array([0.1, 0.6, 0.7])
    batch_results = ensemble.batch_combine_scores(motor_scores, language_scores)
    print(f"Motor scores: {motor_scores}")
    print(f"Language scores: {language_scores}")
    print(f"Combined scores: {batch_results}")

if __name__ == "__main__":
    test_ensemble()
