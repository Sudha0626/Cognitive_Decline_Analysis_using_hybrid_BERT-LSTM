# Memory Guardian - AI Cognitive Decline Detector

## Overview

Memory Guardian is a privacy-focused web application that monitors cognitive health through passive analysis of typing behavior and language patterns. The system uses a dual AI model architecture combining motor skill analysis (LSTM) and text coherence analysis (BiLSTM) to detect potential cognitive decline indicators. All processing occurs locally on the user's device with no cloud uploads, ensuring complete privacy and data control.

The application provides real-time monitoring, historical trend analysis, and an interactive dashboard for visualizing cognitive health metrics over time.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Application Framework
- **Frontend/UI**: Streamlit web framework with Plotly for interactive visualizations
- **Rationale**: Streamlit enables rapid development of data-driven applications with minimal frontend code while providing responsive, interactive dashboards
- **State Management**: Streamlit session state for maintaining user context and monitoring sessions

### AI Model Architecture

**Dual Model System with Ensemble Scoring**

1. **LSTM Motor Skills Model (60% weight)**
   - Architecture: Two-layer LSTM (64→32 units) with dropout regularization
   - Purpose: Analyzes typing speed, errors, backspace usage, pause patterns, and navigation behavior
   - Input: Sequential time-series data (10 timesteps × 5 features)
   - Output: Binary classification (normal vs. concern) with probability scores
   - Rationale: LSTMs excel at capturing temporal patterns in sequential motor behavior data

2. **Bidirectional LSTM Text Coherence Model (40% weight)**
   - Architecture: Embedding layer → BiLSTM (64→32 units) → Dense layers
   - Purpose: Evaluates language coherence, structure, and semantic patterns
   - Input: Tokenized text sequences (max 50-100 tokens)
   - Vocabulary: 2000-10000 most common words with OOV handling
   - Rationale: Bidirectional processing captures both forward and backward context for better coherence detection

3. **Ensemble Scorer**
   - Combines motor (60%) and language (40%) scores into unified cognitive deviation metric
   - Normalizes scores to 0-1 range with interpretive thresholds (Low/Medium/High risk)
   - Provides actionable recommendations based on risk levels

**Design Decisions**:
- Weight distribution (60/40) reflects that motor skills often show earlier degradation than language
- Local processing prioritizes privacy over cloud-based alternatives
- Synthetic data generation for initial training enables deployment without requiring large labeled datasets

### Data Processing Pipeline

1. **Background Monitor**: Captures real-time typing interactions (keystroke timing, errors, navigation)
2. **Data Preprocessor**: Cleans data, handles outliers (IQR-based clipping), extracts features, normalizes using StandardScaler
3. **Feature Engineering**: Converts raw interactions into 5-dimensional feature vectors (typing speed, error count, backspace count, pause duration, navigation jumps)
4. **Sequence Creation**: Builds sliding windows of sequential data for LSTM input

**Rationale**: Pipeline separation enables independent testing, validation, and future enhancement of each component

### Data Storage

**File-Based Local Storage**
- User interaction logs: CSV format (`user_data/interaction_logs.csv`)
- Text content logs: CSV format (`user_data/text_logs.csv`)
- Model weights: TensorFlow SavedModel format (`saved_models/`)
- Configuration: JSON format for model metadata and hyperparameters

**Rationale**: 
- CSV provides human-readable, portable data format
- No database overhead for small-to-medium datasets
- Complete local storage ensures privacy compliance
- Easy data portability and backup

**Trade-offs**:
- Pros: Simple, portable, no database setup, privacy-preserving
- Cons: Not suitable for very large datasets, limited concurrent access, manual indexing

### Training & Evaluation System

- **Synthetic Data Generation**: Creates realistic training data for both motor and text models
- **Train/Validation/Test Split**: 60/20/20 split with stratification for balanced class distribution
- **Callbacks**: Early stopping (patience=10) and model checkpointing based on validation loss
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, AUC-ROC, confusion matrices
- **Rationale**: Comprehensive metrics suite enables detection of model bias and performance issues

### Session Management

- Session-based interaction tracking with reset capabilities
- Real-time metric calculation and aggregation
- Rolling window statistics for trend detection
- Rationale: Enables both immediate feedback and long-term pattern analysis

## External Dependencies

### Core ML/Data Processing Libraries
- **TensorFlow/Keras 2.x**: Deep learning framework for LSTM/BiLSTM models
- **NumPy**: Numerical computations and array operations
- **Pandas**: Structured data manipulation and time-series handling
- **scikit-learn**: Data preprocessing (StandardScaler, MinMaxScaler), train/test splitting, evaluation metrics

### Visualization & UI
- **Streamlit**: Web application framework and interactive dashboard
- **Plotly**: Interactive charts and real-time data visualization
- **Matplotlib/Seaborn**: Static plot generation for evaluation reports

### Data Processing
- **TensorFlow Keras Preprocessing**: Text tokenization and sequence padding

### Development Tools
- **Python 3.11**: Runtime environment
- **uv**: Package management (Replit standard)

**Note**: No external APIs, cloud services, or third-party data services are used. All processing is self-contained.