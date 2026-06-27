# 🧠 Early Cognitive Risk Prediction Using Hybrid BERT-LSTM

An AI-powered web application that passively monitors cognitive patterns through typing behavior and language use, combining a BERT-based text coherence model with an LSTM motor skills model via ensemble fusion — achieving **91.5% accuracy** and **97% AUC**.

---

## 📌 Overview

Early detection of cognitive decline is critical — but clinical assessments are infrequent and expensive. This system takes a different approach: passively observe everyday typing behavior and language patterns to surface early risk signals, all on-device with zero data leaving the user's machine.

**Key capabilities:**
- Passive background monitoring of typing speed, error rate, pause duration, and navigation patterns
- BERT-based Bidirectional LSTM for text coherence analysis
- LSTM Neural Network for motor skill sequence analysis
- Ensemble-based output fusion for robust, calibrated risk scoring
- Privacy-first: 100% local processing — no cloud uploads
- Real-time dashboard with historical trend tracking

---

## 🏗 Model Architecture

```
User Interaction (Typing + Text Input)
              ↓
    ┌─────────────────────┐
    │  Background Monitor │  ← typing speed, errors, pauses, navigation
    └─────────────────────┘
              ↓
   ┌──────────┴──────────┐
   ↓                     ↓
LSTM Model           BiLSTM Model
(Motor Skills)       (Text Coherence)
60% weight           40% weight
   ↓                     ↓
   └──────────┬──────────┘
              ↓
    Ensemble Scorer
    Final Score = 0.6 × Motor + 0.4 × Language
              ↓
    Risk Level Assessment + Dashboard
```

---

## 📊 Results

| Metric | Score |
|---|---|
| Overall Accuracy | **91.5%** |
| AUC (ROC) | **97%** |
| Motor Model Accuracy | ~85–95% |
| Text Coherence Model Accuracy | ~85–95% |

---

## 🎯 Risk Classification

| Score | Risk Level |
|---|---|
| 0.0 – 0.3 | 🟢 Low Risk (Normal) |
| 0.3 – 0.5 | 🟡 Low-Medium Risk |
| 0.5 – 0.7 | 🟠 Medium Risk (Moderate Concern) |
| 0.7 – 1.0 | 🔴 High Risk (Significant Concern) |

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Text Model | Bidirectional LSTM (bert_model.py) |
| Motor Model | LSTM Neural Network (lstm_model.py) |
| Ensemble | Weighted score fusion (ensemble.py) |
| Frontend | Streamlit |
| Deep Learning | TensorFlow / Keras |
| Data Processing | NumPy, Pandas, Scikit-learn |
| Visualization | Plotly, Matplotlib, Seaborn |
| Language | Python 3.11 |

---

## 📁 Project Structure

```
Cognitive_Decline_Analysis_using_hybrid_BERT-LSTM/
├── app.py                  # Main Streamlit application
├── background_monitor.py   # Typing & interaction monitoring
├── data_preprocessing.py   # Feature extraction & data cleaning
├── bert_model.py           # BiLSTM text coherence model
├── lstm_model.py           # LSTM motor skills model
├── ensemble.py             # Ensemble scoring system
├── train.py                # Model training pipeline
├── evaluate.py             # Evaluation metrics & ROC curves
├── utils.py                # Utility functions
└── README.md
```

---

## ⚙️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Sudha0626/Cognitive_Decline_Analysis_using_hybrid_BERT-LSTM.git
cd Cognitive_Decline_Analysis_using_hybrid_BERT-LSTM

# Install dependencies
pip install streamlit tensorflow numpy pandas scikit-learn matplotlib seaborn plotly

# Train the models first
python train.py

# Run the app
streamlit run app.py --server.port 5000
```

---

## 🔒 Privacy & Security

- ✅ All processing is on-device — no data leaves the machine
- ✅ No external API calls or cloud inference
- ✅ Data stored locally in CSV files under user control
- ✅ Privacy-by-design architecture throughout

---

## ⚠️ Disclaimer

This is a research and educational system — not a medical diagnostic tool. Real-world deployment would require validation against clinically verified datasets and review by healthcare professionals.

---

## 👩‍💻 Author

**Raaga Sudha** — [LinkedIn](https://linkedin.com/in/raagasudha06) · [GitHub](https://github.com/Sudha0626)
