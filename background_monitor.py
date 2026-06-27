import time
import numpy as np
from datetime import datetime
from typing import Dict, List
from utils import save_interaction_data, save_text_data, get_timestamp

class BackgroundMonitor:
    def __init__(self):
        self.typing_start_time = None
        self.char_count = 0
        self.error_count = 0
        self.backspace_count = 0
        self.pause_times = []
        self.last_action_time = None
        self.navigation_jumps = 0
        self.current_session = []
        
    def reset_session(self):
        self.typing_start_time = None
        self.char_count = 0
        self.error_count = 0
        self.backspace_count = 0
        self.pause_times = []
        self.last_action_time = None
        self.navigation_jumps = 0
        
    def track_keystroke(self, key: str):
        current_time = time.time()
        
        if self.typing_start_time is None:
            self.typing_start_time = current_time
            
        if self.last_action_time is not None:
            pause = current_time - self.last_action_time
            if pause > 0.5:
                self.pause_times.append(pause)
                
        if key == 'Backspace':
            self.backspace_count += 1
        elif len(key) == 1:
            self.char_count += 1
            
        self.last_action_time = current_time
        
    def track_error(self):
        self.error_count += 1
        
    def track_navigation_jump(self):
        self.navigation_jumps += 1
        
    def calculate_typing_speed(self) -> float:
        if self.typing_start_time is None or self.char_count == 0:
            return 0.0
            
        duration = time.time() - self.typing_start_time
        if duration == 0:
            return 0.0
            
        words = self.char_count / 5.0
        minutes = duration / 60.0
        wpm = words / minutes if minutes > 0 else 0
        
        return wpm
        
    def calculate_avg_pause(self) -> float:
        if len(self.pause_times) == 0:
            return 0.0
        return np.mean(self.pause_times)
        
    def save_session_data(self):
        typing_speed = self.calculate_typing_speed()
        avg_pause = self.calculate_avg_pause()
        
        data = {
            'timestamp': get_timestamp(),
            'typing_speed': round(typing_speed, 2),
            'error_count': self.error_count,
            'backspace_count': self.backspace_count,
            'pause_duration': round(avg_pause, 3),
            'navigation_jumps': self.navigation_jumps
        }
        
        save_interaction_data(data)
        self.reset_session()
        
        return data
        
    def save_text_entry(self, text: str):
        if not text or len(text.strip()) == 0:
            return
            
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        data = {
            'timestamp': get_timestamp(),
            'text_content': text,
            'word_count': word_count,
            'char_count': char_count
        }
        
        save_text_data(data)
        return data

def simulate_typing_session(num_chars: int = 100, error_rate: float = 0.05):
    monitor = BackgroundMonitor()
    
    keys = list('abcdefghijklmnopqrstuvwxyz ')
    
    for i in range(num_chars):
        key = np.random.choice(keys)
        monitor.track_keystroke(key)
        
        if np.random.random() < error_rate:
            monitor.track_error()
            monitor.track_keystroke('Backspace')
            
        time.sleep(np.random.uniform(0.05, 0.3))
        
    if np.random.random() < 0.3:
        monitor.track_navigation_jump()
        
    return monitor.save_session_data()

if __name__ == "__main__":
    print("Simulating typing session...")
    data = simulate_typing_session()
    print(f"Session data: {data}")
