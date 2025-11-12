import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os
import joblib

sys.path.append('src')

def main():
    st.set_page_config(page_title="Model Training - EMI Predict AI", layout="wide")
    
    st.title("🤖 Machine Learning Model Training")
    st.markdown("Train and compare multiple models for EMI prediction")
    
    # Check if source files exist
    st.subheader("🔧 Setup Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Source Files:**")
        required_files = [
            "src/__init__.py",
            "src/config.py", 
            "src/data_preprocessing.py",
            "src/model_training.py", 
            "src/mlflow_utils.py"
        ]
        
        all_files_exist = True
        for file in required_files:
            if os.path.exists(file):
                st.success(f"✅ {file}")
            else:
                st.error(f"❌ {file}")
                all_files_exist = False
    
    with col2:
        st.write("**Dataset:**")
        dataset_path = "data/raw/emi_prediction_dataset.csv"
        if os.path.exists(dataset_path):
            try:
                df = pd.read_csv(dataset_path, dtype={0: str}, low_memory=False)
                st.success(f"✅ Dataset loaded: {df.shape}")
                st.metric("Records", f"{len(df):,}")
                st.metric("Features", len(df.columns))
                
                # Show sample of the dataset
                with st.expander("View Dataset Sample"):
                    st.dataframe(df.head(5), width='stretch')
                    
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
        else:
            st.error("❌ Dataset not found")
            st.info("Please ensure 'emi_prediction_dataset.csv' is in data/raw/ folder")
    
    # Model Training Section
    st.markdown("---")
    st.subheader("🎯 Train Machine Learning Models")
    
    # Training configuration
    st.write("### ⚙️ Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Classification Models to Train:**")
        class_models = st.multiselect(
            "Select classification models:",
            ["Logistic Regression", "Random Forest", "XGBoost", "SVM", "Decision Tree", "Gradient Boosting"],
            default=["Logistic Regression", "Random Forest", "XGBoost"],
            key="class_models"
        )
    
    with col2:
        st.write("**Regression Models to Train:**")
        reg_models = st.multiselect(
            "Select regression models:",
            ["Linear Regression", "Random Forest", "XGBoost", "SVM", "Decision Tree", "Gradient Boosting"],
            default=["Linear Regression", "Random Forest", "XGBoost"],
            key="reg_models"
        )
    
    # Show selected models
    if class_models or reg_models:
        st.info(f"**Selected for training:** Classification: {class_models} | Regression: {reg_models}")
    
    # Start Training Button
    if st.button("🚀 Start Model Training", type="primary", use_container_width=True):
        if not all_files_exist:
            st.error("❌ Cannot start training. Some required files are missing.")
            return
            
        if not class_models and not reg_models:
            st.error("❌ Please select at least one model to train.")
            return
            
        try:
            from src.data_preprocessing import DataPreprocessor
            from src.model_training import ModelTrainer
            from src.config import Config
            
            with st.spinner("🔄 Training models... This may take a few minutes."):
                # Create progress bar and status
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Data Preprocessing
                status_text.text("📊 Step 1/4: Loading and preprocessing data...")
                preprocessor = DataPreprocessor()
                result = preprocessor.full_pipeline(Config.RAW_DATA_PATH)
                
                if result is None:
                    st.error("❌ Data preprocessing failed!")
                    return
                    
                data_dict, features, df_processed = result
                progress_bar.progress(25)
                
                if data_dict is not None:
                    status_text.text("✅ Data preprocessing completed!")
                    
                    # Display preprocessing results
                    st.info(f"**Preprocessing Complete:** {len(features)} features, {len(data_dict['X_train'])} training samples")
                    
                    # Step 2: Initialize Model Trainer with selected models
                    status_text.text("🤖 Step 2/4: Initializing model trainer...")
                    trainer = ModelTrainer(class_models=class_models, reg_models=reg_models)
                    
                    # Step 3: Train Models
                    status_text.text("🎯 Step 3/4: Training models...")
                    training_results = trainer.full_training_pipeline(data_dict)
                    progress_bar.progress(75)
                    
                    # Step 4: Verify and Display Results
                    status_text.text("📊 Step 4/4: Generating results...")
                    progress_bar.progress(100)
                    
                    # Display Results
                    st.subheader("📊 Training Results")
                    
                    results_col1, results_col2 = st.columns(2)
                    
                    with results_col1:
                        if training_results['best_classification_model']:
                            st.metric("Best Classification Model", training_results['best_classification_model'])
                        else:
                            st.error("No classification model trained")
                    
                    with results_col2:
                        if training_results['best_regression_model']:
                            st.metric("Best Regression Model", training_results['best_regression_model'])
                        else:
                            st.error("No regression model trained")
                    
                    # Model Performance Summary
                    st.subheader("📈 Model Performance Summary")
                    
                    perf_col1, perf_col2 = st.columns(2)
                    
                    with perf_col1:
                        st.write("**Trained Models:**")
                        if class_models:
                            st.write("**Classification:**")
                            for model in class_models:
                                status = "✅" if model in training_results['classification_models'] else "❌"
                                st.write(f"{status} {model}")
                        if reg_models:
                            st.write("**Regression:**")
                            for model in reg_models:
                                status = "✅" if model in training_results['regression_models'] else "❌"
                                st.write(f"{status} {model}")
                    
                    with perf_col2:
                        st.write("**Next Steps:**")
                        st.success("""
                        1. **Go to MLflow Dashboard** to compare all model performances
                        2. **Use EMI Calculator** with the trained models
                        3. **Check model artifacts** in the models/ folder
                        4. **View experiment tracking** in mlruns/ folder
                        """)
                    
                    # Save preprocessing artifacts
                    st.info("💾 **Models and preprocessing artifacts saved to:**")
                    st.write(f"- `{Config.MODELS_PATH}` - Trained models")
                    st.write(f"- `{Config.PROCESSED_DATA_PATH}` - Processed data")
                    st.write(f"- `{Config.MLFLOW_TRACKING_URI}` - Experiment tracking")
                        
                else:
                    st.error("❌ Data preprocessing failed! Check your dataset format.")
                    
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())  # Show full traceback for debugging
    
    # Training Information Section
    st.markdown("---")
    st.subheader("📚 Training Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.write("**🤖 What Gets Trained:**")
        st.write("""
        - **6 Classification Models** for EMI eligibility prediction
        - **6 Regression Models** for maximum EMI calculation  
        - **Automatic model selection** based on performance
        - **MLflow experiment tracking** for all runs
        - **Feature importance analysis**
        """)
        
        st.write("**📊 Evaluation Metrics:**")
        st.write("""
        - **Classification:** Accuracy, Precision, Recall, F1-Score
        - **Regression:** RMSE, MAE, R² Score, MAPE
        - **Cross-validation** on validation set
        - **Final evaluation** on test set
        """)
    
    with info_col2:
        st.write("**🔄 Training Process:**")
        st.write("""
        1. **Data Loading & Preprocessing**
        2. **Feature Engineering**
        3. **Train-Test-Validation Split**
        4. **Model Training (Selected Algorithms)**
        5. **Performance Evaluation**
        6. **Model Selection & Saving**
        7. **Experiment Tracking with MLflow**
        """)
        
        st.write("**💡 Expected Results:**")
        st.write("""
        - **Classification Accuracy:** > 90%
        - **Regression RMSE:** < ₹2,000
        - **Feature Importance** charts
        - **Model Comparison** dashboard
        - **Production-ready** models
        """)
    
    # Quick Start Guide
    with st.expander("🚀 Quick Start Guide"):
        st.write("""
        **To train models successfully:**
        
        1. **Ensure all source files** are in the `src/` folder
        2. **Place your dataset** at `data/raw/emi_prediction_dataset.csv`
        3. **Select models** to train from the options above
        4. **Click 'Start Model Training'** and wait for completion
        5. **Check results** in MLflow Dashboard and EMI Calculator
        
        **Required Dataset Columns:**
        - Personal info: age, gender, marital_status, dependents
        - Employment: monthly_income, employment_type, employment_length  
        - Financial: loan_amount, credit_score, total_monthly_obligations
        - Loan details: loan_tenure_months, interest_rate
        
        **Note about SVM:** For large datasets, SVM uses a subset of data for faster training to avoid memory issues.
        """)

if __name__ == "__main__":
    main()