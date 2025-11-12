import pandas as pd
import numpy as np
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Create derived features
        if 'monthly_income' in X_copy.columns and 'total_monthly_obligations' in X_copy.columns:
            X_copy['debt_to_income_ratio'] = (
                X_copy['total_monthly_obligations'] / X_copy['monthly_income']
            )
        
        if 'loan_amount' in X_copy.columns and 'monthly_income' in X_copy.columns:
            X_copy['loan_to_income_ratio'] = (
                X_copy['loan_amount'] / X_copy['monthly_income']
            )
        
        if 'coapplicant_income' in X_copy.columns and 'monthly_income' in X_copy.columns:
            X_copy['coapplicant_income_ratio'] = (
                X_copy['coapplicant_income'] / X_copy['monthly_income']
            )
        
        if 'monthly_income' in X_copy.columns and 'total_monthly_obligations' in X_copy.columns:
            X_copy['income_adequacy_ratio'] = (
                (X_copy['monthly_income'] - X_copy['total_monthly_obligations']) / 
                X_copy['monthly_income']
            )
        
        # Risk scoring features
        if 'credit_score' in X_copy.columns:
            X_copy['credit_score_bucket'] = pd.cut(
                X_copy['credit_score'], 
                bins=[0, 580, 670, 740, 800, 850],
                labels=['Poor', 'Fair', 'Good', 'Very_Good', 'Excellent']
            )
        
        if 'age' in X_copy.columns:
            X_copy['age_group'] = pd.cut(
                X_copy['age'],
                bins=[0, 25, 35, 45, 55, 100],
                labels=['Young', 'Adult', 'Middle_Aged', 'Senior', 'Elderly']
            )
        
        self.feature_names = X_copy.columns.tolist()
        return X_copy
    
    def get_feature_names(self):
        return self.feature_names

def create_interaction_features(df):
    """Create interaction features between key financial variables"""
    df_interaction = df.copy()
    
    # Income * Credit Score interaction
    if 'monthly_income' in df.columns and 'credit_score' in df.columns:
        df_interaction['income_credit_interaction'] = (
            df['monthly_income'] * df['credit_score'] / 1000
        )
    
    # Debt * Age interaction
    if 'debt_to_income_ratio' in df.columns and 'age' in df.columns:
        df_interaction['debt_age_interaction'] = df['debt_to_income_ratio'] * df['age']
    
    # Employment stability indicator
    if 'employment_length' in df.columns and 'monthly_income' in df.columns:
        df_interaction['employment_stability'] = (
            df['employment_length'] * df['monthly_income'] / 1000
        )
    
    return df_interaction