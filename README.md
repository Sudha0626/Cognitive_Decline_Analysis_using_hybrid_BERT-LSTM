# Memory Guardian - AI Cognitive Decline Detector

A comprehensive AI-powered web application for monitoring and analyzing cognitive patterns through passive observation of typing behavior and language use.

## Features

- **Passive Background Monitoring**: Tracks typing speed, errors, navigation flow, and text input
- **Dual AI Model System**:
  - LSTM Neural Network for motor skill analysis (60% weight)
  - Bidirectional LSTM for text coherence analysis (40% weight)
- **Ensemble Scoring**: Combines both analyses into a comprehensive cognitive deviation score
- **Interactive Dashboard**: Real-time visualization of cognitive health metrics
- **Historical Analysis**: Track trends over time
- **100% Local Processing**: All data and analysis stays on your device
- **Privacy-First Design**: No cloud uploads, complete user control

## Project Structure

```
.
├── app.py                    # Main Streamlit application
├── background_monitor.py     # Typing and interaction monitoring
├── data_preprocessing.py     # Data cleaning and feature extraction
├── lstm_model.py            # LSTM motor skills model
├── bert_model.py            # BiLSTM text coherence model
├── ensemble.py              # Ensemble scoring system
├── train.py                 # Model training script
├── evaluate.py              # Model evaluation and metrics
├── utils.py                 # Utility functions
├── saved_models/            # Trained model storage
└── user_data/               # User interaction and text logs
```

## Installation

### Prerequisites

- Python 3.11
- pip or uv package manager

### Install Dependencies

```bash
# Using uv (recommended in Replit)
uv sync

# Or using pip
pip install streamlit tensorflow numpy pandas scikit-learn matplotlib seaborn plotly
```

## Usage

### 1. Train the Models

Before using the application, train the AI models:

```bash
python train.py
```

This will:
- Generate synthetic training data
- Train the LSTM motor skills model
- Train the BiLSTM text coherence model
- Save both models to `saved_models/`
- Display training metrics and performance

### 2. Run the Application

```bash
streamlit run app.py --server.port 5000
```

The application will be available at `http://localhost:5000`

### 3. Collect Data

Navigate to the "Data Collection" page to:
- Simulate typing sessions
- Add text entries
- Generate sample datasets for testing

### 4. View Dashboard

The Dashboard provides:
- Real-time cognitive deviation scores
- Typing speed trends
- Error rate analysis
- Language coherence metrics
- Risk level assessment

## How It Works

### Motor Skills Analysis (60% weight)

The LSTM model analyzes time-series sequences of:
- **Typing Speed**: Words per minute (WPM)
- **Error Count**: Number of typing errors
- **Backspace Frequency**: Correction patterns
- **Pause Durations**: Time between keystrokes
- **Navigation Jumps**: Cursor movement patterns

### Text Coherence Analysis (40% weight)

The Bidirectional LSTM model evaluates:
- Text coherence and structure
- Vocabulary usage patterns
- Sentence formation quality
- Word count and complexity

### Ensemble Scoring

The final cognitive deviation score is calculated as:
```
Final Score = 0.6 × Motor Score + 0.4 × Language Score
```

Risk levels:
- **0.0 - 0.3**: Low Risk (Normal)
- **0.3 - 0.5**: Low-Medium Risk (Slight Deviation)
- **0.5 - 0.7**: Medium Risk (Moderate Concern)
- **0.7 - 1.0**: High Risk (Significant Concern)

## Model Performance

Both models are trained on synthetic data and achieve:
- Motor Model: ~85-95% accuracy, AUC ~0.90+
- Text Model: ~85-95% accuracy, AUC ~0.90+
- Ensemble: Combined performance with balanced contribution

## Evaluation

To evaluate model performance:

```bash
python evaluate.py
```

This generates:
- Accuracy, Precision, Recall, F1-Score metrics
- ROC curves and AUC scores
- Confusion matrices
- Score comparison visualizations
- All plots saved to `saved_models/`

## API Reference

### BackgroundMonitor

```python
from background_monitor import BackgroundMonitor

monitor = BackgroundMonitor()
monitor.track_keystroke('a')
monitor.track_error()
data = monitor.save_session_data()
```

### LSTMMotorModel

```python
from lstm_model import LSTMMotorModel

model = LSTMMotorModel(sequence_length=10, n_features=5)
model.build_model()
model.train(X_train, y_train, X_val, y_val)
model.save_model()
score = model.predict_single(sequence)
```

### SimpleTextCoherenceModel

```python
from bert_model import SimpleTextCoherenceModel

model = SimpleTextCoherenceModel(vocab_size=2000, max_length=50)
model.build_model()
model.train(texts_train, y_train, texts_val, y_val)
model.save_model()
score = model.predict_single("Your text here")
```

### EnsembleScorer

```python
from ensemble import EnsembleScorer

ensemble = EnsembleScorer(motor_weight=0.6, language_weight=0.4)
final_score = ensemble.combine_scores(motor_score, language_score)
interpretation = ensemble.interpret_score(final_score)
```

## Data Format

### Interaction Logs (`user_data/interaction_logs.csv`)

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Time of session |
| typing_speed | float | Words per minute |
| error_count | int | Number of errors |
| backspace_count | int | Backspace presses |
| pause_duration | float | Average pause (seconds) |
| navigation_jumps | int | Cursor movements |

### Text Logs (`user_data/text_logs.csv`)

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Time of entry |
| text_content | string | Entered text |
| word_count | int | Number of words |
| char_count | int | Number of characters |

## Privacy & Security

- ✅ All processing happens locally on your device
- ✅ No data is sent to external servers
- ✅ All data stored in local CSV files
- ✅ Models saved locally in `saved_models/`
- ✅ Complete user control over data

## Limitations

- This is a demonstration/educational system
- Not intended for medical diagnosis
- Should be validated by healthcare professionals
- Trained on synthetic data for demonstration purposes
- Real-world deployment requires validated medical datasets

## Technical Stack

- **Frontend**: Streamlit
- **Deep Learning**: TensorFlow/Keras
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Language**: Python 3.11

## License

Educational/Research Use Only

## Support

For questions, issues, or contributions, please consult the documentation or contact your healthcare provider for medical advice.

## Version

Version 1.0.0 - November 2024
