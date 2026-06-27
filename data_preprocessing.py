import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List
from utils import load_interaction_data, load_text_data

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        
    def clean_interaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df = df.dropna()
        
        numeric_cols = ['typing_speed', 'error_count', 'backspace_count', 
                       'pause_duration', 'navigation_jumps']
        
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        if len(df) == 0:
            return np.array([]).reshape(0, 5)
            
        features = []
        feature_cols = ['typing_speed', 'error_count', 'backspace_count', 
                       'pause_duration', 'navigation_jumps']
        
        for col in feature_cols:
            if col in df.columns:
                features.append(df[col].values)
            else:
                features.append(np.zeros(len(df)))
                
        return np.column_stack(features)
    
    def normalize_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        if len(X) == 0:
            return X
            
        if fit:
            self.scaler.fit(X)
            self.fitted = True
            
        if self.fitted:
            return self.scaler.transform(X)
        else:
            if len(X) > 0:
                self.scaler.fit(X)
                self.fitted = True
                return self.scaler.transform(X)
            return X
    
    def create_sequences(self, X: np.ndarray, sequence_length: int = 10) -> np.ndarray:
        if len(X) < sequence_length:
            padding_needed = sequence_length - len(X)
            padding = np.zeros((padding_needed, X.shape[1]))
            X = np.vstack([padding, X])
            
        sequences = []
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i:i + sequence_length])
            
        return np.array(sequences)
    
    def prepare_motor_data(self, sequence_length: int = 10, fit: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
        df = load_interaction_data()
        
        if len(df) == 0:
            return np.array([]).reshape(0, sequence_length, 5), df
            
        df_clean = self.clean_interaction_data(df)
        
        features = self.extract_features(df_clean)
        
        features_normalized = self.normalize_features(features, fit=fit)
        
        sequences = self.create_sequences(features_normalized, sequence_length)
        
        return sequences, df_clean
    
    def prepare_text_data(self) -> Tuple[List[str], pd.DataFrame]:
        df = load_text_data()
        
        if len(df) == 0:
            return [], df
            
        df = df.dropna(subset=['text_content'])
        
        texts = df['text_content'].tolist()
        
        return texts, df
    
    def extract_text_features(self, text: str) -> dict:
        if not text or len(text.strip()) == 0:
            return {
                'word_count': 0,
                'char_count': 0,
                'avg_word_length': 0,
                'unique_words': 0,
                'sentence_count': 0
            }
            
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        avg_word_length = char_count / word_count if word_count > 0 else 0
        unique_words = len(set(words))
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'avg_word_length': avg_word_length,
            'unique_words': unique_words,
            'sentence_count': max(1, sentence_count)
        }

def test_preprocessor():
    from utils import create_sample_data
    
    create_sample_data(50)
    
    preprocessor = DataPreprocessor()
    
    sequences, df = preprocessor.prepare_motor_data(sequence_length=10)
    print(f"Motor sequences shape: {sequences.shape}")
    print(f"Interaction data records: {len(df)}")
    
    texts, text_df = preprocessor.prepare_text_data()
    print(f"Text records: {len(texts)}")
    
    if len(texts) > 0:
        features = preprocessor.extract_text_features(texts[0])
        print(f"Sample text features: {features}")

if __name__ == "__main__":
    test_preprocessor()
