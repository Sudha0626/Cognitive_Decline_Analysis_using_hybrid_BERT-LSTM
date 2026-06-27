import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional

def ensure_directories():
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('user_data', exist_ok=True)

def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_interaction_data(filepath: str = 'user_data/interaction_logs.csv') -> pd.DataFrame:
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return pd.DataFrame(columns=['timestamp', 'typing_speed', 'error_count', 'backspace_count', 
                                  'pause_duration', 'navigation_jumps'])

def load_text_data(filepath: str = 'user_data/text_logs.csv') -> pd.DataFrame:
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return pd.DataFrame(columns=['timestamp', 'text_content', 'word_count', 'char_count'])

def save_interaction_data(data: Dict, filepath: str = 'user_data/interaction_logs.csv'):
    df = load_interaction_data(filepath)
    new_row = pd.DataFrame([data])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(filepath, index=False)

def save_text_data(data: Dict, filepath: str = 'user_data/text_logs.csv'):
    df = load_text_data(filepath)
    new_row = pd.DataFrame([data])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(filepath, index=False)

def calculate_statistics(df: pd.DataFrame, column: str, window: int = 10) -> Dict:
    if len(df) == 0 or column not in df.columns:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'recent_mean': 0}
    
    recent_data = df[column].tail(window)
    return {
        'mean': df[column].mean(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'recent_mean': recent_data.mean() if len(recent_data) > 0 else 0
    }

def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(max_val, score))

def get_model_path(model_name: str) -> str:
    return os.path.join('saved_models', f'{model_name}.keras')

def model_exists(model_name: str) -> bool:
    return os.path.exists(get_model_path(model_name))

def save_config(config: Dict, filepath: str = 'saved_models/config.json'):
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(filepath: str = 'saved_models/config.json') -> Dict:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def create_sample_data(n_samples: int = 100):
    ensure_directories()
    
    np.random.seed(42)
    interaction_data = []
    text_data = []
    
    for i in range(n_samples):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        typing_speed = np.random.normal(60, 10)
        error_count = np.random.poisson(2)
        backspace_count = np.random.poisson(3)
        pause_duration = np.random.exponential(0.5)
        navigation_jumps = np.random.poisson(1)
        
        interaction_data.append({
            'timestamp': timestamp,
            'typing_speed': max(0, typing_speed),
            'error_count': error_count,
            'backspace_count': backspace_count,
            'pause_duration': pause_duration,
            'navigation_jumps': navigation_jumps
        })
        
        sample_texts = [
            "Hello, how are you doing today?",
            "The weather is beautiful this morning.",
            "I need to remember to buy groceries later.",
            "Working on an important project right now.",
            "This is a test of the typing monitoring system."
        ]
        
        text_content = np.random.choice(sample_texts)
        word_count = len(text_content.split())
        char_count = len(text_content)
        
        text_data.append({
            'timestamp': timestamp,
            'text_content': text_content,
            'word_count': word_count,
            'char_count': char_count
        })
    
    pd.DataFrame(interaction_data).to_csv('user_data/interaction_logs.csv', index=False)
    pd.DataFrame(text_data).to_csv('user_data/text_logs.csv', index=False)
    
    print(f"Created {n_samples} sample records in user_data/")

if __name__ == "__main__":
    create_sample_data()
