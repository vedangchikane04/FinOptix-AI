import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False
        
    def load_data(self, file_path):
        """Load and initial data inspection"""
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Data loaded successfully: {df.shape}")
            
            # Check core features status
            status = Config.get_core_features_status(df.columns)
            print("\n🔍 Core Features Status:")
            for feature, status_msg in status.items():
                print(f"   {feature}: {status_msg}")
                
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def validate_numeric_columns(self, df, columns):
        """Ensure specified columns are numeric"""
        df_clean = df.copy()
        for col in columns:
            if col and col in df_clean.columns:
                original_type = df_clean[col].dtype
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                if df_clean[col].isnull().any():
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                print(f"✅ Converted {col} from {original_type} to {df_clean[col].dtype}")
        return df_clean
    
    def safe_divide(self, numerator, denominator, default=0):
        """Safe division that handles type issues and division by zero"""
        try:
            # Ensure both are numeric arrays
            num_array = pd.to_numeric(numerator, errors='coerce').fillna(0).values
            den_array = pd.to_numeric(denominator, errors='coerce').fillna(1).values  # Use 1 to avoid division by zero
            
            # Avoid division by zero
            result = np.where(den_array != 0, num_array / den_array, default)
            return result
        except Exception as e:
            print(f"❌ Division error: {e}")
            return np.full(len(numerator), default)
    
    def clean_data(self, df):
        """Data cleaning and validation"""
        # Create a copy
        df_clean = df.copy()
        
        # Convert ALL potential numeric columns first
        potential_numeric_cols = [
            'monthly_income', 'annual_income', 'income', 'salary',
            'credit_score', 'credit_rating', 'cibil_score', 'credit',
            'existing_emi', 'monthly_obligations', 'obligations', 'existing_loan',
            'loan_amount', 'requested_loan_amount', 'loan',
            'age', 'experience', 'tenure'
        ]
        
        for col in potential_numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Now handle missing values
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        # Fill numerical missing values with median
        for col in numerical_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        print(f"✅ Data cleaned: {df_clean.shape}")
        
        # Debug info
        print("\n📊 Data types after cleaning:")
        print(df_clean.dtypes)
        print(f"\n📊 Missing values after cleaning:")
        print(df_clean.isnull().sum())
        
        return df_clean
    
    def create_synthetic_loan_amount(self, df):
        """Create synthetic loan amount based on income if not available"""
        income_col = Config.get_income_column(df.columns)
        
        if income_col:
            # Ensure income is numeric
            df_temp = df.copy()
            df_temp[income_col] = pd.to_numeric(df_temp[income_col], errors='coerce')
            df_temp[income_col].fillna(df_temp[income_col].median(), inplace=True)
            
            # Create loan amount as 2-5 times monthly income (typical range)
            np.random.seed(42)  # For reproducibility
            income_multiplier = np.random.uniform(2, 5, len(df))
            df['synthetic_loan_amount'] = df_temp[income_col] * income_multiplier
            print("✅ Created synthetic loan amount based on income")
            return 'synthetic_loan_amount'
        else:
            # If no income, create random loan amounts
            np.random.seed(42)
            df['synthetic_loan_amount'] = np.random.randint(50000, 5000000, len(df))
            print("⚠️ Created random synthetic loan amounts")
            return 'synthetic_loan_amount'
    
    def create_target_variables(self, df):
        """Create EMI eligibility classification and max EMI regression targets"""
        df_processed = df.copy()
        
        # Find core columns
        loan_col = Config.get_loan_amount_column(df.columns)
        income_col = Config.get_income_column(df.columns)
        credit_col = Config.get_credit_score_column(df.columns)
        obligation_col = Config.get_obligation_column(df.columns)
        
        # Validate and ensure numeric types for critical columns
        critical_cols = [col for col in [income_col, credit_col, obligation_col, loan_col] if col]
        df_processed = self.validate_numeric_columns(df_processed, critical_cols)
        
        # Create synthetic loan amount if missing
        if not loan_col:
            loan_col = self.create_synthetic_loan_amount(df_processed)
        
        # Check if we have minimum required features
        if not income_col or not credit_col:
            raise ValueError("❌ CRITICAL: Dataset must contain either 'monthly_income' or 'credit_score' columns")
        
        # Calculate debt-to-income ratio using safe division
        if obligation_col and obligation_col in df_processed.columns:
            # Use actual obligations if available
            df_processed['debt_to_income_ratio'] = self.safe_divide(
                df_processed[obligation_col], 
                df_processed[income_col],
                default=0.5  # Default to high ratio if calculation fails
            )
            print(f"✅ Using '{obligation_col}' for debt-to-income calculation")
        else:
            # If no obligation column, create a simple ratio based on loan amount
            df_processed['debt_to_income_ratio'] = self.safe_divide(
                df_processed[loan_col], 
                (df_processed[income_col] * 12),  # Annualized
                default=0.5
            )
            print("⚠️ No obligation column found. Using loan amount for ratio calculation.")
        
        # Ensure credit score is properly formatted
        df_processed[credit_col] = pd.to_numeric(df_processed[credit_col], errors='coerce')
        df_processed[credit_col].fillna(df_processed[credit_col].median(), inplace=True)
        
        # Create EMI Eligibility Classification (3 classes)
        conditions = [
            # Eligible: Good financial health
            (df_processed['debt_to_income_ratio'] <= 0.3) & 
            (df_processed[credit_col] >= 700),
            
            # High Risk: Moderate financial health
            ((df_processed['debt_to_income_ratio'] > 0.3) & (df_processed['debt_to_income_ratio'] <= 0.5)) |
            ((df_processed[credit_col] >= 600) & (df_processed[credit_col] < 700)),
            
            # Not Eligible: Poor financial health
            (df_processed['debt_to_income_ratio'] > 0.5) | 
            (df_processed[credit_col] < 600)
        ]
        
        choices = ['Eligible', 'High_Risk', 'Not_Eligible']
        df_processed['emi_eligibility'] = np.select(conditions, choices, default='Not_Eligible')
        
        # Create Max Monthly EMI Regression Target
        if obligation_col and obligation_col in df_processed.columns:
            # Use actual obligations if available
            max_affordable = df_processed[income_col] * 0.4 - df_processed[obligation_col]
        else:
            # Estimate based on income if no obligations data
            max_affordable = df_processed[income_col] * 0.3  # More conservative
        
        # Ensure max_affordable is reasonable
        max_affordable = np.maximum(500, np.minimum(50000, max_affordable))
        df_processed['max_monthly_emi'] = max_affordable
        
        print("✅ Target variables created successfully")
        print(f"📊 EMI Eligibility Distribution:\n{df_processed['emi_eligibility'].value_counts()}")
        print(f"📊 Max Monthly EMI Stats: min={df_processed['max_monthly_emi'].min():.2f}, "
              f"max={df_processed['max_monthly_emi'].max():.2f}, "
              f"mean={df_processed['max_monthly_emi'].mean():.2f}")
        
        return df_processed
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        # Get categorical features dynamically based on available columns
        categorical_features = Config.get_categorical_features(df.columns)
        
        for col in categorical_features:
            if col in df_encoded.columns:
                # Handle NaN values before encoding
                df_encoded[col] = df_encoded[col].fillna('Unknown')
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                print(f"✅ Encoded categorical feature: {col}")
        
        self.is_fitted = True
        return df_encoded
    
    def prepare_features(self, df):
        """Prepare final feature set"""
        # Get features based on available columns
        numerical_features = Config.get_numerical_features(df.columns)
        categorical_features = Config.get_categorical_features(df.columns)
        
        # Find core columns for derived features
        loan_col = Config.get_loan_amount_column(df.columns)
        income_col = Config.get_income_column(df.columns)
        obligation_col = Config.get_obligation_column(df.columns)
        
        # Create synthetic loan amount if missing (it should already be created)
        if not loan_col and 'synthetic_loan_amount' in df.columns:
            loan_col = 'synthetic_loan_amount'
        
        # Create derived features using safe division
        derived_features = []
        
        if 'debt_to_income_ratio' not in df.columns:
            if obligation_col and obligation_col in df.columns and income_col and income_col in df.columns:
                df['debt_to_income_ratio'] = self.safe_divide(df[obligation_col], df[income_col])
                derived_features.append('debt_to_income_ratio')
            elif loan_col and loan_col in df.columns and income_col and income_col in df.columns:
                df['debt_to_income_ratio'] = self.safe_divide(df[loan_col], (df[income_col] * 12))
                derived_features.append('debt_to_income_ratio')
        
        if loan_col and loan_col in df.columns and income_col and income_col in df.columns:
            df['loan_to_income_ratio'] = self.safe_divide(df[loan_col], df[income_col])
            derived_features.append('loan_to_income_ratio')
        
        # Combine all available features
        feature_columns = numerical_features + categorical_features + derived_features
        
        # Select only available features
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Ensure target columns exist
        if Config.TARGET_CLASSIFICATION not in df.columns:
            raise ValueError(f"❌ Missing target column: {Config.TARGET_CLASSIFICATION}")
        if Config.TARGET_REGRESSION not in df.columns:
            raise ValueError(f"❌ Missing target column: {Config.TARGET_REGRESSION}")
        
        X = df[available_features]
        y_class = df[Config.TARGET_CLASSIFICATION]
        y_reg = df[Config.TARGET_REGRESSION]
        
        print(f"✅ Using {len(available_features)} features: {available_features}")
        print(f"✅ Classification target: {y_class.unique()}")
        print(f"✅ Regression target range: {y_reg.min():.2f} to {y_reg.max():.2f}")
        
        return X, y_class, y_reg, available_features
    
    def split_data(self, X, y_class, y_reg):
        """Split data into train, validation, and test sets"""
        # First split: separate test set
        X_temp, X_test, y_class_temp, y_class_test, y_reg_temp, y_reg_test = train_test_split(
            X, y_class, y_reg, 
            test_size=Config.TEST_SIZE, 
            random_state=Config.RANDOM_STATE,
            stratify=y_class
        )
        
        # Second split: separate validation set from temp
        val_size_adjusted = Config.VALIDATION_SIZE / (1 - Config.TEST_SIZE)
        X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val = train_test_split(
            X_temp, y_class_temp, y_reg_temp,
            test_size=val_size_adjusted,
            random_state=Config.RANDOM_STATE,
            stratify=y_class_temp
        )
        
        print(f"✅ Data split completed:")
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Validation set: {X_val.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_class_train': y_class_train, 'y_class_val': y_class_val, 'y_class_test': y_class_test,
            'y_reg_train': y_reg_train, 'y_reg_val': y_reg_val, 'y_reg_test': y_reg_test
        }
    
    def save_processed_data(self, data_dict, path):
        """Save processed data"""
        os.makedirs(path, exist_ok=True)
        
        for key, value in data_dict.items():
            if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                value.to_csv(os.path.join(path, f"{key}.csv"), index=False)
        
        # Save preprocessing artifacts
        joblib.dump(self.scaler, os.path.join(path, 'scaler.joblib'))
        joblib.dump(self.label_encoders, os.path.join(path, 'label_encoders.joblib'))
        
        print(f"✅ Processed data saved to: {path}")
    
    def full_pipeline(self, file_path):
        """Complete data preprocessing pipeline"""
        try:
            print("🚀 Starting data preprocessing pipeline...")
            
            # Load data
            df = self.load_data(file_path)
            if df is None:
                return None
            
            # Debug: Check initial data
            print("\n🔍 Initial Data Info:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Data types:\n{df.dtypes}")
            
            # Clean data
            df_clean = self.clean_data(df)
            
            # Create target variables
            df_processed = self.create_target_variables(df_clean)
            
            # Encode categorical variables
            df_encoded = self.encode_categorical(df_processed)
            
            # Prepare features
            X, y_class, y_reg, features = self.prepare_features(df_encoded)
            
            # Split data
            data_dict = self.split_data(X, y_class, y_reg)
            
            # Scale numerical features
            numerical_cols = [col for col in Config.get_numerical_features(df.columns) if col in features]
            if numerical_cols:  # Only scale if there are numerical columns
                print(f"🔧 Scaling numerical features: {numerical_cols}")
                data_dict['X_train'][numerical_cols] = self.scaler.fit_transform(data_dict['X_train'][numerical_cols])
                data_dict['X_val'][numerical_cols] = self.scaler.transform(data_dict['X_val'][numerical_cols])
                data_dict['X_test'][numerical_cols] = self.scaler.transform(data_dict['X_test'][numerical_cols])
            
            print("✅ Data preprocessing completed successfully!")
            return data_dict, features, df_processed
            
        except Exception as e:
            print(f"❌ Error in preprocessing pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None


# Example usage and testing
if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    
    # Example with test data path
    test_file_path = "data/raw/emi_prediction_dataset.csv"  # Adjust path as needed
    
    # Run full pipeline
    result = preprocessor.full_pipeline(test_file_path)
    
    if result:
        data_dict, features, processed_df = result
        print(f"\n🎉 Pipeline completed! Features used: {len(features)}")
        print(f"📊 Processed data shape: {processed_df.shape}")
    else:
        print("❌ Pipeline failed!")