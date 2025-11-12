import os
from datetime import datetime

class Config:
    # Paths
    RAW_DATA_PATH = "data/raw/emi_prediction_dataset.csv"
    PROCESSED_DATA_PATH = "data/processed/"
    MODELS_PATH = "models/"
    
    # MLflow
    MLFLOW_TRACKING_URI = "mlruns"
    EXPERIMENT_NAME = f"EMI_Predict_AI_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Model Parameters
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    RANDOM_STATE = 42
    
    # Core required features - if missing, we'll create synthetic ones
    REQUIRED_CORE_FEATURES = ['monthly_income', 'credit_score']
    
    # Feature Engineering - FLEXIBLE: Will use available columns or create synthetic ones
    POTENTIAL_LOAN_AMOUNT_COLUMNS = [
        'loan_amount', 'loan_amt', 'amount', 'principal', 
        'loan_value', 'requested_amount', 'loan'
    ]
    
    POTENTIAL_OBLIGATION_COLUMNS = [
        'total_monthly_obligations', 'monthly_obligations', 
        'existing_emi', 'current_emi', 'monthly_debt', 
        'debt_amount', 'outstanding_balance', 'total_debt'
    ]
    
    POTENTIAL_INCOME_COLUMNS = [
        'monthly_income', 'income', 'salary', 'monthly_salary',
        'gross_income', 'net_income'
    ]
    
    POTENTIAL_CREDIT_SCORE_COLUMNS = [
        'credit_score', 'credit_rating', 'cibil_score', 'score'
    ]
    
    # These will be dynamically determined based on available columns
    @classmethod
    def get_core_features_status(cls, df_columns):
        """Check which core features are available"""
        status = {}
        
        # Check loan amount
        loan_col = cls.get_loan_amount_column(df_columns)
        status['loan_amount'] = loan_col if loan_col else "MISSING - will create synthetic"
        
        # Check monthly income
        income_col = cls.get_income_column(df_columns)
        status['monthly_income'] = income_col if income_col else "MISSING - CRITICAL"
        
        # Check credit score
        credit_col = cls.get_credit_score_column(df_columns)
        status['credit_score'] = credit_col if credit_col else "MISSING - CRITICAL"
        
        # Check obligations
        obligation_col = cls.get_obligation_column(df_columns)
        status['obligations'] = obligation_col if obligation_col else "MISSING - will estimate"
        
        return status
    
    @classmethod
    def get_loan_amount_column(cls, df_columns):
        """Find the loan amount column in the dataset"""
        for col in cls.POTENTIAL_LOAN_AMOUNT_COLUMNS:
            if col in df_columns:
                return col
        return None
    
    @classmethod
    def get_income_column(cls, df_columns):
        """Find the income column in the dataset"""
        for col in cls.POTENTIAL_INCOME_COLUMNS:
            if col in df_columns:
                return col
        return None
    
    @classmethod
    def get_credit_score_column(cls, df_columns):
        """Find the credit score column in the dataset"""
        for col in cls.POTENTIAL_CREDIT_SCORE_COLUMNS:
            if col in df_columns:
                return col
        return None
    
    @classmethod
    def get_obligation_column(cls, df_columns):
        """Find the obligation column in the dataset"""
        for col in cls.POTENTIAL_OBLIGATION_COLUMNS:
            if col in df_columns:
                return col
        return None
    
    @classmethod
    def get_numerical_features(cls, df_columns):
        """Dynamically get numerical features from available columns"""
        # First, find our core columns
        loan_col = cls.get_loan_amount_column(df_columns)
        income_col = cls.get_income_column(df_columns)
        credit_col = cls.get_credit_score_column(df_columns)
        
        # Start with core features we found
        available_features = []
        if income_col:
            available_features.append(income_col)
        if credit_col:
            available_features.append(credit_col)
        if loan_col:
            available_features.append(loan_col)
        
        # Add other potential numerical features
        other_potential_features = [
            'age', 'loan_tenure_months', 'interest_rate', 'emi_amount',
            'employment_length', 'dependents', 'coapplicant_income',
            'num_active_loans', 'num_credit_cards', 'previous_defaults',
            'num_inquiries_last_6m', 'delinquency_30_days', 'credit_history_length'
        ]
        
        for feature in other_potential_features:
            if feature in df_columns:
                available_features.append(feature)
        
        return available_features
    
    @classmethod
    def get_categorical_features(cls, df_columns):
        """Dynamically get categorical features from available columns"""
        potential_features = ['gender', 'marital_status', 'employment_type', 'purpose']
        return [feature for feature in potential_features if feature in df_columns]
    
    # Static definitions (fallback)
    NUMERICAL_FEATURES = ['monthly_income', 'credit_score']
    CATEGORICAL_FEATURES = ['gender', 'marital_status', 'employment_type']
    DERIVED_FEATURES = ['debt_to_income_ratio', 'loan_to_income_ratio']
    TARGET_CLASSIFICATION = 'emi_eligibility'
    TARGET_REGRESSION = 'max_monthly_emi'