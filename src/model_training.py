import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import joblib
import warnings
warnings.filterwarnings('ignore')
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config


class ModelTrainer:
    def __init__(self, experiment_name=None, class_models=None, reg_models=None):
        # Use provided experiment name or default from Config
        self.experiment_name = experiment_name or Config.EXPERIMENT_NAME
        self.selected_class_models = class_models or []
        self.selected_reg_models = reg_models or []
        self.classification_models = {}
        self.regression_models = {}
        self.best_classification_model = None
        self.best_regression_model = None
        self.best_classification_name = None
        self.best_regression_name = None
        self.label_encoder = LabelEncoder()
        
        # Set up MLflow
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.experiment_name)
    
    def initialize_models(self):
        """Initialize only the selected classification and regression models"""
        # Classification Models mapping
        all_class_models = {
            'Logistic Regression': LogisticRegression(
                random_state=Config.RANDOM_STATE, 
                max_iter=1000,
                C=1.0
            ),
            'Random Forest': RandomForestClassifier(
                random_state=Config.RANDOM_STATE, 
                n_estimators=100,
                max_depth=10
            ),
            'XGBoost': XGBClassifier(
                random_state=Config.RANDOM_STATE, 
                n_estimators=100,
                learning_rate=0.1,
                eval_metric='logloss'
            ),
            'SVM': SVC(
                random_state=Config.RANDOM_STATE, 
                probability=True,
                kernel='rbf',
                C=1.0,
                gamma='scale'
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=Config.RANDOM_STATE,
                max_depth=10
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=Config.RANDOM_STATE, 
                n_estimators=100,
                learning_rate=0.1
            )
        }
        
        # Regression Models mapping
        all_reg_models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                random_state=Config.RANDOM_STATE, 
                n_estimators=100,
                max_depth=10
            ),
            'XGBoost': XGBRegressor(
                random_state=Config.RANDOM_STATE, 
                n_estimators=100,
                learning_rate=0.1
            ),
            'SVM': SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale'
            ),
            'Decision Tree': DecisionTreeRegressor(
                random_state=Config.RANDOM_STATE,
                max_depth=10
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                random_state=Config.RANDOM_STATE, 
                n_estimators=100,
                learning_rate=0.1
            )
        }
        
        # Only initialize selected classification models
        self.classification_models = {}
        for model_name in self.selected_class_models:
            if model_name in all_class_models:
                self.classification_models[model_name] = all_class_models[model_name]
        
        # Only initialize selected regression models
        self.regression_models = {}
        for model_name in self.selected_reg_models:
            if model_name in all_reg_models:
                self.regression_models[model_name] = all_reg_models[model_name]
    
    def evaluate_classification(self, y_true, y_pred, y_pred_proba=None):
        """Evaluate classification model performance"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except:
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def evaluate_regression(self, y_true, y_pred):
        """Evaluate regression model performance"""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100
        }
        return metrics
    
    def train_classification_models(self, X_train, y_train, X_val, y_val):
        """Train selected classification models with MLflow tracking"""
        if not self.classification_models:
            st.warning("⚠️ No classification models selected for training")
            return None
        
        st.write("## 🎯 Training Classification Models...")
        
        # Encode target variable
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        
        best_score = 0
        best_model_name = None
        
        for model_name, model in self.classification_models.items():
            st.write(f"**Training {model_name}...**")
            
            with mlflow.start_run(run_name=f"Classification_{model_name}", nested=True):
                try:
                    # Log model name as parameter
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("model_type", "classification")
                    
                    # Special handling for SVM with large datasets
                    if model_name == 'SVM':
                        # Use smaller subset for SVM to avoid memory issues
                        sample_size = min(10000, len(X_train))
                        X_train_sub = X_train[:sample_size]
                        y_train_sub = y_train_encoded[:sample_size]
                        model.fit(X_train_sub, y_train_sub)
                    else:
                        model.fit(X_train, y_train_encoded)
                    
                    # Predictions
                    y_pred = model.predict(X_val)
                    y_pred_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
                    
                    # Evaluate
                    metrics = self.evaluate_classification(y_val_encoded, y_pred, y_pred_proba)
                    
                    # Log parameters and metrics
                    mlflow.log_params(model.get_params())
                    mlflow.log_metrics(metrics)
                    
                    # Log model
                    mlflow.sklearn.log_model(model, f"classification_{model_name.lower().replace(' ', '_')}")
                    
                    # Update best model
                    if metrics['f1_score'] > best_score:
                        best_score = metrics['f1_score']
                        best_model_name = model_name
                        self.best_classification_model = model
                        self.best_classification_name = model_name
                    
                    st.success(f"✅ {model_name} completed")
                    
                except Exception as e:
                    st.error(f"❌ Error training {model_name}: {str(e)}")
                    continue
        
        if best_model_name:
            st.success(f"🏆 Best Classification Model Selected: {best_model_name}")
        else:
            st.error("❌ No classification models were successfully trained")
            
        return best_model_name
    
    def train_regression_models(self, X_train, y_train, X_val, y_val):
        """Train selected regression models with MLflow tracking"""
        if not self.regression_models:
            st.warning("⚠️ No regression models selected for training")
            return None
        
        st.write("## 📈 Training Regression Models...")
        
        best_score = float('inf')  # Lower RMSE is better
        best_model_name = None
        
        for model_name, model in self.regression_models.items():
            st.write(f"**Training {model_name}...**")
            
            with mlflow.start_run(run_name=f"Regression_{model_name}", nested=True):
                try:
                    # Log model name as parameter
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("model_type", "regression")
                    
                    # Special handling for SVM with large datasets
                    if model_name == 'SVM':
                        # Use smaller subset for SVM to avoid memory issues
                        sample_size = min(10000, len(X_train))
                        X_train_sub = X_train[:sample_size]
                        y_train_sub = y_train[:sample_size]
                        model.fit(X_train_sub, y_train_sub)
                    else:
                        model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_val)
                    
                    # Evaluate
                    metrics = self.evaluate_regression(y_val, y_pred)
                    
                    # Log parameters and metrics
                    mlflow.log_params(model.get_params())
                    mlflow.log_metrics(metrics)
                    
                    # Log model
                    mlflow.sklearn.log_model(model, f"regression_{model_name.lower().replace(' ', '_')}")
                    
                    # Update best model
                    if metrics['rmse'] < best_score:
                        best_score = metrics['rmse']
                        best_model_name = model_name
                        self.best_regression_model = model
                        self.best_regression_name = model_name
                    
                    st.success(f"✅ {model_name} completed")
                    
                except Exception as e:
                    st.error(f"❌ Error training {model_name}: {str(e)}")
                    continue
        
        if best_model_name:
            st.success(f"🏆 Best Regression Model Selected: {best_model_name}")
        else:
            st.error("❌ No regression models were successfully trained")
            
        return best_model_name
    
    def save_models(self, path=Config.MODELS_PATH):
        """Save trained models and preprocessing artifacts"""
        os.makedirs(path, exist_ok=True)
        
        try:
            if self.best_classification_model is not None:
                joblib.dump(self.best_classification_model, os.path.join(path, 'best_classification_model.joblib'))
                joblib.dump(self.label_encoder, os.path.join(path, 'label_encoder.joblib'))
            else:
                st.warning("⚠️ No classification model to save")
            
            if self.best_regression_model is not None:
                joblib.dump(self.best_regression_model, os.path.join(path, 'best_regression_model.joblib'))
            else:
                st.warning("⚠️ No regression model to save")
            
            # Save model info
            model_info = {
                'best_classification_model': self.best_classification_name,
                'best_regression_model': self.best_regression_name,
                'selected_class_models': self.selected_class_models,
                'selected_reg_models': self.selected_reg_models,
                'timestamp': pd.Timestamp.now()
            }
            joblib.dump(model_info, os.path.join(path, 'model_info.joblib'))
            
            st.success("💾 Best models saved successfully!")
            
        except Exception as e:
            st.error(f"❌ Error saving models: {e}")
    
    def full_training_pipeline(self, data_dict):
        """Complete model training pipeline"""
        # Initialize only selected models
        self.initialize_models()
        
        # Train classification models
        best_class_model = self.train_classification_models(
            data_dict['X_train'], data_dict['y_class_train'],
            data_dict['X_val'], data_dict['y_class_val']
        )
        
        # Train regression models
        best_reg_model = self.train_regression_models(
            data_dict['X_train'], data_dict['y_reg_train'],
            data_dict['X_val'], data_dict['y_reg_val']
        )
        
        # Save models
        self.save_models()
        
        return {
            'best_classification_model': self.best_classification_name,
            'best_regression_model': self.best_regression_name,
            'classification_models': list(self.classification_models.keys()),
            'regression_models': list(self.regression_models.keys())
        }