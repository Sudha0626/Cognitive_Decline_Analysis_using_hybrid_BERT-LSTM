import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

from utils import (ensure_directories, load_interaction_data, load_text_data, 
                  calculate_statistics, model_exists, create_sample_data)
from background_monitor import BackgroundMonitor, simulate_typing_session
from data_preprocessing import DataPreprocessor
from lstm_model import LSTMMotorModel
from bert_model import SimpleTextCoherenceModel
from ensemble import EnsembleScorer

st.set_page_config(
    page_title="Memory Guardian",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

ensure_directories()

if 'monitor' not in st.session_state:
    st.session_state.monitor = BackgroundMonitor()

if 'text_buffer' not in st.session_state:
    st.session_state.text_buffer = ""

def load_models():
    motor_model = None
    text_model = None
    ensemble = EnsembleScorer(motor_weight=0.6, language_weight=0.4)
    
    if model_exists('lstm_motor_model'):
        motor_model = LSTMMotorModel(sequence_length=10, n_features=5)
        try:
            motor_model.load_model()
        except Exception as e:
            st.warning(f"Could not load motor model: {e}")
            motor_model = None
    
    if model_exists('text_coherence_model'):
        text_model = SimpleTextCoherenceModel(vocab_size=2000, max_length=50)
        try:
            text_model.load_model()
        except Exception as e:
            st.warning(f"Could not load text model: {e}")
            text_model = None
    
    return motor_model, text_model, ensemble

def home_page():
    st.title("🧠 Memory Guardian")
    st.subheader("AI Cognitive Decline Detector")
    
    st.markdown("""
    ### Welcome to Memory Guardian
    
    Memory Guardian is an advanced AI-powered system designed to monitor and analyze cognitive patterns 
    through passive observation of your typing behavior and language use.
    
    #### How It Works
    
    1. **Passive Monitoring** 📊
       - Tracks typing speed, errors, and navigation patterns
       - Analyzes text coherence and language patterns
       - All processing happens locally on your device
    
    2. **AI Analysis** 🤖
       - **LSTM Neural Network**: Analyzes motor skills and interaction patterns
       - **Text Coherence Model**: Evaluates language quality and coherence
       - **Ensemble System**: Combines both analyses for comprehensive assessment
    
    3. **Privacy-First Design** 🔒
       - 100% local processing - no cloud uploads
       - All data stored on your device
       - You maintain complete control of your information
    
    #### Features
    
    - **Real-time Monitoring**: Track typing patterns as you type
    - **Cognitive Scoring**: Get instant feedback on cognitive health indicators
    - **Historical Analysis**: View trends over time
    - **Detailed Insights**: Understand motor skills vs language patterns
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Motor Analysis", "LSTM Neural Network", "60% Weight")
    with col2:
        st.metric("Language Analysis", "Text Coherence Model", "40% Weight")
    with col3:
        st.metric("Privacy", "100% Local", "No Cloud")
    
    st.markdown("---")
    
    st.info("👈 Use the sidebar to navigate between pages. Start by collecting data or viewing the Dashboard!")
    
    if not model_exists('lstm_motor_model') or not model_exists('text_coherence_model'):
        st.warning("""
        ⚠️ **Models not found!** 
        
        To use the cognitive analysis features, you need to train the AI models first.
        Run the training script from the terminal:
        
        ```bash
        python train.py
        ```
        
        This will train both the LSTM motor model and text coherence model.
        """)

def dashboard_page():
    st.title("📊 Cognitive Health Dashboard")
    
    motor_model, text_model, ensemble = load_models()
    
    df_interaction = load_interaction_data()
    df_text = load_text_data()
    
    if len(df_interaction) == 0 and len(df_text) == 0:
        st.warning("No data available yet. Please collect some data first!")
        
        if st.button("Generate Sample Data"):
            create_sample_data(100)
            st.success("Sample data created! Refresh the page to see the dashboard.")
            st.rerun()
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Motor Skills Analysis")
        
        if len(df_interaction) > 0:
            typing_stats = calculate_statistics(df_interaction, 'typing_speed')
            error_stats = calculate_statistics(df_interaction, 'error_count')
            
            st.metric(
                "Avg Typing Speed",
                f"{typing_stats['recent_mean']:.1f} WPM",
                delta=f"{typing_stats['recent_mean'] - typing_stats['mean']:.1f}"
            )
            
            st.metric(
                "Recent Errors",
                f"{error_stats['recent_mean']:.1f}",
                delta=f"{error_stats['recent_mean'] - error_stats['mean']:.1f}",
                delta_color="inverse"
            )
            
            fig_typing = px.line(
                df_interaction.tail(50),
                x=df_interaction.tail(50).index,
                y='typing_speed',
                title='Typing Speed Trend (Last 50 Sessions)',
                labels={'typing_speed': 'WPM', 'index': 'Session'}
            )
            fig_typing.add_hline(y=typing_stats['mean'], line_dash="dash", 
                               annotation_text="Average", line_color="red")
            st.plotly_chart(fig_typing, use_container_width=True)
            
            fig_errors = px.bar(
                df_interaction.tail(20),
                x=df_interaction.tail(20).index,
                y='error_count',
                title='Error Count (Last 20 Sessions)',
                labels={'error_count': 'Errors', 'index': 'Session'}
            )
            st.plotly_chart(fig_errors, use_container_width=True)
        else:
            st.info("No motor skills data available.")
    
    with col2:
        st.subheader("Language Analysis")
        
        if len(df_text) > 0:
            word_stats = calculate_statistics(df_text, 'word_count')
            
            st.metric(
                "Avg Words per Entry",
                f"{word_stats['recent_mean']:.1f}",
                delta=f"{word_stats['recent_mean'] - word_stats['mean']:.1f}"
            )
            
            st.metric(
                "Total Entries",
                len(df_text)
            )
            
            fig_words = px.line(
                df_text.tail(50),
                x=df_text.tail(50).index,
                y='word_count',
                title='Word Count Trend (Last 50 Entries)',
                labels={'word_count': 'Words', 'index': 'Entry'}
            )
            st.plotly_chart(fig_words, use_container_width=True)
            
            if len(df_text) >= 10:
                recent_texts = df_text.tail(10)
                fig_text_dist = px.histogram(
                    recent_texts,
                    x='word_count',
                    title='Word Count Distribution (Last 10 Entries)',
                    nbins=10
                )
                st.plotly_chart(fig_text_dist, use_container_width=True)
        else:
            st.info("No language data available.")
    
    st.markdown("---")
    st.subheader("🎯 Cognitive Deviation Score")
    
    if motor_model is not None and text_model is not None and len(df_interaction) > 0 and len(df_text) > 0:
        try:
            preprocessor = DataPreprocessor()
            sequences, _ = preprocessor.prepare_motor_data(sequence_length=10, fit=False)
            
            if len(sequences) > 0:
                motor_score = motor_model.predict_single(sequences[-1])
            else:
                motor_score = 0.3
            
            texts, _ = preprocessor.prepare_text_data()
            if len(texts) > 0:
                text_score = text_model.predict_single(texts[-1])
            else:
                text_score = 0.3
            
            final_score = ensemble.combine_scores(motor_score, text_score)
            interpretation = ensemble.interpret_score(final_score)
            contributions = ensemble.get_component_contributions(motor_score, text_score)
            
            col1, col2, col3 = st.columns([2, 2, 3])
            
            with col1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=interpretation['percentage'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Cognitive Deviation"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': interpretation['color']},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 50], 'color': "lightyellow"},
                            {'range': [50, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                st.metric("Risk Level", interpretation['risk_level'])
                st.metric("Status", interpretation['status'])
                st.metric("Motor Score", f"{motor_score:.3f}")
                st.metric("Language Score", f"{text_score:.3f}")
            
            with col3:
                st.markdown(f"**Score Breakdown:**")
                st.progress(contributions['motor_contribution'], 
                          text=f"Motor Contribution: {contributions['motor_contribution']:.3f} ({contributions['motor_percentage']:.1f}%)")
                st.progress(contributions['language_contribution'],
                          text=f"Language Contribution: {contributions['language_contribution']:.3f} ({contributions['language_percentage']:.1f}%)")
                
                st.markdown("---")
                st.info(f"**Recommendation:** {interpretation['recommendation']}")
                
        except Exception as e:
            st.error(f"Error calculating cognitive score: {e}")
            st.info("Try generating more data or retraining the models.")
    else:
        if motor_model is None or text_model is None:
            st.warning("""
            ⚠️ **AI Models Not Trained**
            
            To enable cognitive scoring, you need to train the AI models first.
            
            **Steps to train models:**
            1. Open a terminal/shell
            2. Run: `python train.py`
            3. Wait for training to complete (~2-5 minutes)
            4. Refresh this page to see cognitive scores
            
            The models will be saved to `saved_models/` and automatically loaded on next visit.
            """)
        else:
            st.info("Collect some interaction and text data to see cognitive scores. Use the Data Collection page to get started!")

def history_page():
    st.title("📈 Historical Analysis")
    
    df_interaction = load_interaction_data()
    df_text = load_text_data()
    
    if len(df_interaction) == 0 and len(df_text) == 0:
        st.warning("No historical data available.")
        return
    
    tab1, tab2 = st.tabs(["Motor Skills History", "Language History"])
    
    with tab1:
        st.subheader("Motor Skills Data")
        
        if len(df_interaction) > 0:
            st.dataframe(df_interaction.tail(50), use_container_width=True)
            
            st.download_button(
                label="Download Motor Data (CSV)",
                data=df_interaction.to_csv(index=False),
                file_name="motor_skills_data.csv",
                mime="text/csv"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_typing_hist = px.histogram(
                    df_interaction,
                    x='typing_speed',
                    title='Typing Speed Distribution',
                    nbins=30
                )
                st.plotly_chart(fig_typing_hist, use_container_width=True)
            
            with col2:
                fig_error_hist = px.histogram(
                    df_interaction,
                    x='error_count',
                    title='Error Count Distribution',
                    nbins=20
                )
                st.plotly_chart(fig_error_hist, use_container_width=True)
        else:
            st.info("No motor skills data available.")
    
    with tab2:
        st.subheader("Language Data")
        
        if len(df_text) > 0:
            st.dataframe(df_text.tail(50), use_container_width=True)
            
            st.download_button(
                label="Download Language Data (CSV)",
                data=df_text.to_csv(index=False),
                file_name="language_data.csv",
                mime="text/csv"
            )
            
            fig_word_hist = px.histogram(
                df_text,
                x='word_count',
                title='Word Count Distribution',
                nbins=30
            )
            st.plotly_chart(fig_word_hist, use_container_width=True)
        else:
            st.info("No language data available.")

def data_collection_page():
    st.title("📝 Data Collection")
    
    models_trained = model_exists('lstm_motor_model') and model_exists('text_coherence_model')
    
    if not models_trained:
        st.info("""
        💡 **Getting Started:**
        1. Generate sample data below (for testing)
        2. Train the AI models by running `python train.py` in terminal
        3. View the Dashboard to see cognitive analysis
        """)
    
    st.markdown("""
    Use this page to simulate data collection for testing the system.
    In a real application, this would happen automatically in the background.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Simulate Typing Session")
        
        num_chars = st.slider("Number of characters", 50, 500, 100)
        error_rate = st.slider("Error rate", 0.0, 0.5, 0.05, 0.01)
        
        if st.button("Simulate Typing"):
            with st.spinner("Simulating typing session..."):
                data = simulate_typing_session(num_chars, error_rate)
                st.success("Typing session simulated!")
                st.json(data)
    
    with col2:
        st.subheader("Add Text Entry")
        
        text_input = st.text_area("Enter text:", height=150)
        
        if st.button("Save Text Entry"):
            if text_input.strip():
                st.session_state.monitor.save_text_entry(text_input)
                st.success("Text entry saved!")
                st.json({
                    'text': text_input,
                    'word_count': len(text_input.split()),
                    'char_count': len(text_input)
                })
            else:
                st.error("Please enter some text first.")
    
    st.markdown("---")
    
    if st.button("Generate Sample Dataset"):
        with st.spinner("Generating sample data..."):
            create_sample_data(100)
            st.success("Generated 100 sample records!")
            st.rerun()

def about_page():
    st.title("ℹ️ About Memory Guardian")
    
    st.markdown("""
    ### System Architecture
    
    Memory Guardian uses a sophisticated ensemble of deep learning models to analyze cognitive patterns:
    
    #### 1. Motor Skills Analysis (60% weight)
    - **Model**: LSTM (Long Short-Term Memory) Neural Network
    - **Input**: Time-series sequences of typing behavior
    - **Features Analyzed**:
        - Typing speed (Words Per Minute)
        - Error count and backspace frequency
        - Pause durations between keystrokes
        - Navigation jumps and cursor movements
    
    #### 2. Language Pattern Analysis (40% weight)
    - **Model**: Bidirectional LSTM Text Coherence Model
    - **Note**: Uses BiLSTM architecture (not BERT) due to dependency constraints
    - **Input**: Text entries and written content
    - **Features Analyzed**:
        - Text coherence and structure
        - Vocabulary usage and diversity
        - Sentence formation patterns
        - Word count and complexity
    
    #### 3. Ensemble Scoring System
    - Combines motor (60%) and language (40%) scores
    - Outputs a cognitive deviation score (0-1)
    - Provides risk level assessment and recommendations
    
    ### Privacy & Security
    
    - ✅ **100% Local Processing**: All analysis happens on your device
    - ✅ **No Cloud Uploads**: Your data never leaves your computer
    - ✅ **Open Source Models**: Transparent AI architecture
    - ✅ **User Control**: You own and control all your data
    
    ### Technical Stack
    
    - **Frontend**: Streamlit
    - **Deep Learning**: TensorFlow/Keras
    - **Data Processing**: NumPy, Pandas, Scikit-learn
    - **Visualization**: Plotly, Matplotlib, Seaborn
    
    ### Model Performance
    
    Both models are trained on synthetic data to demonstrate the system's capabilities.
    For real-world use, models should be trained on validated medical datasets.
    
    ### Limitations
    
    - This is a demonstration system and should not be used for medical diagnosis
    - Results should be validated by healthcare professionals
    - Cognitive health is complex and requires comprehensive clinical assessment
    
    ### Version Information
    
    - **Version**: 1.0.0
    - **Last Updated**: November 2024
    - **License**: Educational/Research Use
    """)
    
    st.markdown("---")
    st.info("💡 For questions or support, consult the documentation or contact your healthcare provider.")

def main():
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Dashboard", "Data Collection", "History", "About"]
    )
    
    st.sidebar.markdown("---")
    
    df_interaction = load_interaction_data()
    df_text = load_text_data()
    
    st.sidebar.metric("Motor Records", len(df_interaction))
    st.sidebar.metric("Text Records", len(df_text))
    
    models_trained = model_exists('lstm_motor_model') and model_exists('text_coherence_model')
    st.sidebar.metric("Models Trained", "Yes ✅" if models_trained else "No ❌")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Actions")
    
    if st.sidebar.button("Clear All Data"):
        if st.sidebar.checkbox("Confirm deletion"):
            if os.path.exists('user_data/interaction_logs.csv'):
                os.remove('user_data/interaction_logs.csv')
            if os.path.exists('user_data/text_logs.csv'):
                os.remove('user_data/text_logs.csv')
            st.sidebar.success("Data cleared!")
            st.rerun()
    
    if page == "Home":
        home_page()
    elif page == "Dashboard":
        dashboard_page()
    elif page == "Data Collection":
        data_collection_page()
    elif page == "History":
        history_page()
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()
