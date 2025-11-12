import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append('src')

def fix_dataframe_for_display(df):
    """
    Fix dataframe to avoid Arrow serialization errors
    """
    if df is None or df.empty:
        return df
        
    df_fixed = df.copy()
    
    # Convert all object columns to string
    for col in df_fixed.columns:
        if df_fixed[col].dtype == 'object':
            df_fixed[col] = df_fixed[col].astype(str)
    
    return df_fixed

# Try to import modules with error handling
try:
    from src.config import Config
    config_loaded = True
except ImportError as e:
    st.error(f"Config import failed: {e}")
    config_loaded = False
    # Fallback config
    class Config:
        RAW_DATA_PATH = "data/raw/emi_prediction_dataset.csv"
        NUMERICAL_FEATURES = ['age', 'monthly_income', 'loan_amount', 'credit_score', 'total_monthly_obligations']
        CATEGORICAL_FEATURES = ['gender', 'marital_status', 'employment_type']

# Page configuration
st.set_page_config(
    page_title="EMI Predict AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class EMIPredictApp:
    def __init__(self):
        self.loaded_models = {}
    
    def home_page(self):
        """Home page with overview and dashboard"""
        st.markdown('<div class="main-header">🏦 EMI Predict AI - Intelligent Financial Risk Assessment</div>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Dataset", "emi_prediction_dataset.csv")
        
        with col2:
            st.metric("Target Accuracy", "> 90%")
        
        with col3:
            st.metric("Risk Categories", "3 Levels")
        
        st.markdown("---")
        
        # Quick overview
        st.subheader("📊 Platform Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Dual ML Problem Solving:**
            - Classification: EMI Eligibility Prediction
            - Regression: Maximum EMI Amount Calculation
            """)
            
            st.success("""
            **Key Features:**
            - Real-time Financial Risk Assessment
            - Comprehensive Data Analysis
            - Interactive EMI Calculator
            - Data-driven Insights
            """)
        
        with col2:
            st.warning("""
            **Business Impact:**
            - 80% Reduction in Manual Processing
            - Automated Risk Assessment
            - Better Loan Decision Making
            - Standardized Evaluation
            """)
            
            st.error("""
            **Technical Stack:**
            - Python, Streamlit, Plotly
            - Pandas, NumPy, Scikit-learn
            - Machine Learning Models
            """)
        
        # Dataset overview
        dataset_path = getattr(Config, 'RAW_DATA_PATH', 'data/raw/emi_prediction_dataset.csv')
        
        if os.path.exists(dataset_path):
            try:
                # FIXED: CSV loading with proper type handling
                df = pd.read_csv(dataset_path, low_memory=False)
                
                # Fix the first column if it has mixed types
                if len(df.columns) > 0:
                    first_col = df.columns[0]
                    df[first_col] = df[first_col].astype(str)
                
                st.subheader("📈 Dataset Overview")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Dataset Info:**")
                    st.write(f"Shape: {df.shape}")
                    st.write(f"Columns: {len(df.columns)}")
                    st.write(f"Records: {len(df):,}")
                    
                    # Show data types
                    st.write("**Data Types:**")
                    dtype_counts = df.dtypes.value_counts()
                    for dtype, count in dtype_counts.items():
                        st.write(f"- {dtype}: {count} columns")
                
                with col2:
                    st.write("**Sample Data:**")
                    # FIXED: Use fixed dataframe for display
                    display_df = fix_dataframe_for_display(df.head(8))
                    st.dataframe(display_df, width='stretch')
                
                # Quick visualizations
                st.subheader("🚀 Quick Insights")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    if 'age' in df.columns:
                        fig_age = px.histogram(df, x='age', title='Age Distribution', 
                                             nbins=20, color_discrete_sequence=['#1f77b4'])
                        st.plotly_chart(fig_age, use_container_width=True)
                
                with viz_col2:
                    if 'monthly_income' in df.columns:
                        fig_income = px.histogram(df, x='monthly_income', title='Income Distribution',
                                                nbins=20, color_discrete_sequence=['#ff7f0e'])
                        st.plotly_chart(fig_income, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
        else:
            st.error(f"❌ Dataset not found at: {dataset_path}")
            st.info("Please make sure your 'emi_prediction_dataset.csv' is in the data/raw/ folder")
    
    def eda_page(self):
        """Exploratory Data Analysis Page"""
        st.header("📊 Exploratory Data Analysis")
        st.markdown("Comprehensive analysis of your EMI prediction dataset")
        
        dataset_path = getattr(Config, 'RAW_DATA_PATH', 'data/raw/emi_prediction_dataset.csv')
        
        if not os.path.exists(dataset_path):
            st.error("Dataset not found. Please check the file path.")
            return
        
        try:
            # FIXED: CSV loading with proper type handling
            df = pd.read_csv(dataset_path, low_memory=False)
            
            # Fix the first column if it has mixed types
            if len(df.columns) > 0:
                first_col = df.columns[0]
                df[first_col] = df[first_col].astype(str)
                
            st.success(f"✅ Dataset loaded successfully: {df.shape[0]:,} records, {df.shape[1]} features")
            
            # Basic statistics
            st.subheader("📋 Dataset Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Data sample and info
            tab1, tab2, tab3 = st.tabs(["Data Sample", "Column Info", "Descriptive Stats"])
            
            with tab1:
                st.write("**First 10 Records:**")
                # FIXED: Use fixed dataframe for display
                display_df1 = fix_dataframe_for_display(df.head(10))
                st.dataframe(display_df1, width='stretch')
            
            with tab2:
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes,
                    'Missing Values': df.isnull().sum(),
                    'Unique Values': df.nunique()
                })
                # FIXED: Use fixed dataframe for display
                display_df2 = fix_dataframe_for_display(col_info)
                st.dataframe(display_df2, width='stretch')
            
            with tab3:
                # FIXED: Use fixed dataframe for display
                display_df3 = fix_dataframe_for_display(df.describe())
                st.dataframe(display_df3, width='stretch')
            
            # Visualizations
            st.subheader("📈 Data Visualizations")
            
            # Numerical features
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numerical_cols) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Numerical Distributions**")
                    selected_num = st.selectbox("Select numerical column:", numerical_cols)
                    if selected_num:
                        fig = px.histogram(df, x=selected_num, 
                                         title=f'Distribution of {selected_num}',
                                         nbins=30,
                                         color_discrete_sequence=['#2E86AB'])
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**Box Plots**")
                    if len(numerical_cols) > 0:
                        box_col = st.selectbox("Select column for box plot:", numerical_cols, key='box')
                        if box_col:
                            fig = px.box(df, y=box_col, title=f'Box Plot of {box_col}',
                                       color_discrete_sequence=['#A23B72'])
                            st.plotly_chart(fig, use_container_width=True)
            
            # Categorical features
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            if len(categorical_cols) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Categorical Distributions**")
                    selected_cat = st.selectbox("Select categorical column:", categorical_cols)
                    if selected_cat:
                        value_counts = df[selected_cat].value_counts()
                        fig = px.bar(x=value_counts.index, y=value_counts.values,
                                   title=f'Distribution of {selected_cat}',
                                   labels={'x': selected_cat, 'y': 'Count'},
                                   color_discrete_sequence=['#F18F01'])
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**Pie Charts**")
                    if len(categorical_cols) > 0:
                        pie_col = st.selectbox("Select column for pie chart:", categorical_cols, key='pie')
                        if pie_col:
                            value_counts = df[pie_col].value_counts()
                            fig = px.pie(values=value_counts.values, names=value_counts.index,
                                       title=f'Distribution of {pie_col}')
                            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix
            if len(numerical_cols) > 1:
                st.subheader("🔗 Correlation Analysis")
                
                # Select columns for correlation
                selected_corr_cols = st.multiselect(
                    "Select columns for correlation matrix:",
                    numerical_cols,
                    default=numerical_cols[:6] if len(numerical_cols) >= 6 else numerical_cols
                )
                
                if len(selected_corr_cols) > 1:
                    corr_matrix = df[selected_corr_cols].corr()
                    fig = px.imshow(corr_matrix,
                                  title="Correlation Matrix",
                                  color_continuous_scale='RdBu_r',
                                  aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show highest correlations
                    st.write("**Top Correlations:**")
                    corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
                    # Remove self-correlations and duplicates
                    corr_pairs = corr_pairs[corr_pairs < 0.999]
                    top_correlations = corr_pairs.head(10)
                    
                    for (col1, col2), value in top_correlations.items():
                        st.write(f"- {col1} ↔ {col2}: {value:.3f}")
            
            # Outlier detection
            st.subheader("🔍 Outlier Analysis")
            
            if len(numerical_cols) > 0:
                outlier_col = st.selectbox("Select column for outlier detection:", numerical_cols, key='outlier')
                if outlier_col:
                    Q1 = df[outlier_col].quantile(0.25)
                    Q3 = df[outlier_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Outliers", len(outliers))
                        st.metric("Outlier Percentage", f"{(len(outliers)/len(df))*100:.2f}%")
                    
                    with col2:
                        fig = px.box(df, y=outlier_col, title=f'Box Plot - {outlier_col}',
                                   color_discrete_sequence=['#C73E1D'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if len(outliers) > 0:
                        with st.expander("View Outlier Data"):
                            # FIXED: Use fixed dataframe for display
                            display_df4 = fix_dataframe_for_display(outliers)
                            st.dataframe(display_df4, width='stretch')
                
        except Exception as e:
            st.error(f"Error in data analysis: {e}")
    
    def prediction_page(self):
        """EMI Prediction and Risk Assessment Page"""
        st.header("🎯 EMI Calculator & Risk Assessment")
        st.markdown("Calculate EMI and assess financial risk in real-time")
    
        st.subheader("Enter Applicant Details")
     
        col1, col2, col3 = st.columns(3)
    
        with col1:
           st.markdown("#### 👤 Personal Information")
           age = st.slider("Age", 18, 70, 35)
           gender = st.selectbox("Gender", ["Male", "Female", "Other"])
           marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
           dependents = st.slider("Number of Dependents", 0, 10, 1)
    
        with col2:
           st.markdown("#### 💼 Employment & Income")
           employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business Owner", "Other"])
           monthly_income = st.number_input("Monthly Income (₹)", 10000, 500000, 50000, step=5000)
           coapplicant_income = st.number_input("Co-applicant Income (₹)", 0, 500000, 0, step=5000)
           employment_length = st.slider("Employment Length (years)", 0, 40, 5)
    
        with col3:
           st.markdown("#### 🏦 Loan & Financial Details")
           loan_amount = st.number_input("Loan Amount (₹)", 10000, 5000000, 500000, step=10000)
           loan_tenure_months = st.slider("Loan Tenure (months)", 6, 84, 36)
           interest_rate = st.slider("Interest Rate (%)", 5.0, 20.0, 10.0, step=0.5)
           credit_score = st.slider("Credit Score", 300, 850, 650)
           total_monthly_obligations = st.number_input("Total Monthly Obligations (₹)", 0, 100000, 10000, step=1000)
    
        # Additional financial details
        st.subheader("📊 Additional Financial Information")
    
        col4, col5 = st.columns(2)
    
        with col4:
           num_active_loans = st.slider("Number of Active Loans", 0, 10, 0)
           num_credit_cards = st.slider("Number of Credit Cards", 0, 10, 2)
           outstanding_balance = st.number_input("Outstanding Balance (₹)", 0, 1000000, 0, step=10000)
    
        with col5:
           previous_defaults = st.slider("Previous Defaults", 0, 5, 0)
           num_inquiries_last_6m = st.slider("Credit Inquiries (Last 6 months)", 0, 10, 1)
           delinquency_30_days = st.slider("30+ Days Delinquencies", 0, 10, 0)
           credit_history_length = st.slider("Credit History Length (years)", 0, 30, 5)
    
        # Calculate EMI and Assess Risk
        if st.button("🚀 Calculate EMI & Assess Risk", type="primary"):
            with st.spinner("Analyzing financial risk profile..."):
                # Calculate EMI
                monthly_interest = interest_rate / 12 / 100
                emi_amount = (loan_amount * monthly_interest * 
                          (1 + monthly_interest) ** loan_tenure_months) / ((1 + monthly_interest) ** loan_tenure_months - 1)
            
                # Calculate financial ratios
                total_income = monthly_income + coapplicant_income
                debt_to_income = total_monthly_obligations / total_income if total_income > 0 else 0
                emi_to_income = emi_amount / total_income if total_income > 0 else 0
                total_debt_ratio = (total_monthly_obligations + emi_amount) / total_income if total_income > 0 else 0
            
                # Display results
                st.success("### 📊 Risk Assessment Results")
            
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
            
                with col1:
                    st.metric("Monthly EMI", f"₹{emi_amount:,.2f}")
            
                with col2:
                    st.metric("Total Interest", f"₹{(emi_amount * loan_tenure_months - loan_amount):,.2f}")
            
                with col3:
                    st.metric("Total Payment", f"₹{(emi_amount * loan_tenure_months):,.2f}")
            
                with col4:
                    st.metric("Debt-to-Income", f"{debt_to_income:.1%}")
            
                # Risk assessment logic
                risk_factors = []
                if debt_to_income > 0.4:
                    risk_factors.append("High debt-to-income ratio (>40%)")
                if emi_to_income > 0.5:
                    risk_factors.append("High EMI-to-income ratio (>50%)")
                if total_debt_ratio > 0.6:
                    risk_factors.append("Total debt ratio too high (>60%)")
                if credit_score < 600:
                    risk_factors.append("Low credit score (<600)")
                if previous_defaults > 0:
                    risk_factors.append(f"Has {previous_defaults} previous default(s)")
                if age < 25:
                    risk_factors.append("Young applicant (higher risk)")
                if employment_length < 2:
                    risk_factors.append("Short employment history (<2 years)")
            
                # Determine eligibility
                risk_score = len(risk_factors)
            
                # FIXED: Delta color handling
                if risk_score == 0:
                    eligibility = "✅ Eligible"
                    recommendation = "Excellent financial profile. Strong candidate for loan approval with best rates."
                    risk_level = "Low Risk"
                    delta_color = "normal"  # Green for positive
                elif risk_score <= 2:
                    eligibility = "⚠️ Conditional Approval"
                    recommendation = "Moderate risk profile. Consider approval with slightly higher interest rates or reduced loan amount."
                    risk_level = "Medium Risk"
                    delta_color = None  # No delta for neutral
                else:
                    eligibility = "❌ Not Eligible"
                    recommendation = "High risk profile. Loan approval not recommended at this time."
                    risk_level = "High Risk"
                    delta_color = "inverse"  # Red for negative
            
                st.subheader("📋 Financial Health Analysis")
            
                analysis_col1, analysis_col2 = st.columns(2)
            
                with analysis_col1:
                    # FIXED: Conditional delta display
                    if delta_color:
                        st.metric("Eligibility Status", eligibility, delta=risk_level, delta_color=delta_color)
                    else:
                        st.metric("Eligibility Status", eligibility, delta=risk_level)
                
                    st.write("**Key Financial Ratios:**")
                    st.write(f"• Debt-to-Income: {debt_to_income:.1%}")
                    st.write(f"• EMI-to-Income: {emi_to_income:.1%}")
                    st.write(f"• Total Debt Ratio: {total_debt_ratio:.1%}")
                    st.write(f"• Credit Score: {credit_score}")
                
                    st.write("**Recommendation:**")
                    if delta_color == "normal":
                        st.success(recommendation)
                    elif delta_color is None:
                        st.warning(recommendation)
                    else:
                        st.error(recommendation)
            
                with analysis_col2:
                    # Risk factors
                    if risk_factors:
                        st.warning("**🚨 Identified Risk Factors:**")
                        for factor in risk_factors:
                            st.write(f"• {factor}")
                    else:
                        st.success("**✅ No significant risk factors detected**")
                
                    # Affordability analysis
                    st.write("**💡 Affordability Insights:**")
                    affordable_emi = total_income * 0.4 - total_monthly_obligations
                    affordability_ratio = emi_amount / affordable_emi if affordable_emi > 0 else 1
                
                    st.write(f"• Maximum affordable EMI: ₹{max(0, affordable_emi):,.2f}")
                    st.write(f"• Current EMI is {affordability_ratio:.1%} of maximum affordable")
                
                    if affordability_ratio > 1:
                        st.error("EMI exceeds affordable limit!")
                    elif affordability_ratio > 0.8:
                        st.warning("EMI is close to affordable limit")
                    else:
                        st.success("EMI is within affordable range")
            
                # Loan summary
                st.subheader("📄 Loan Summary")
            
                summary_col1, summary_col2 = st.columns(2)
            
                with summary_col1:
                    st.write("**Loan Details:**")
                    st.write(f"• Principal Amount: ₹{loan_amount:,.2f}")
                    st.write(f"• Interest Rate: {interest_rate}% p.a.")
                    st.write(f"• Loan Tenure: {loan_tenure_months} months ({loan_tenure_months/12:.1f} years)")
                    st.write(f"• Monthly EMI: ₹{emi_amount:,.2f}")
            
                with summary_col2:
                    st.write("**Financial Overview:**")
                    st.write(f"• Total Interest Payable: ₹{(emi_amount * loan_tenure_months - loan_amount):,.2f}")
                    st.write(f"• Total Repayment: ₹{(emi_amount * loan_tenure_months):,.2f}")
                    st.write(f"• Effective Interest Rate: {((emi_amount * loan_tenure_months / loan_amount - 1) * 100 / (loan_tenure_months/12)):.1f}% p.a.")

    def mlflow_dashboard_page(self):
        """MLflow Experiment Tracking Dashboard"""
        st.header("📈 MLflow Experiment Tracking")
        st.markdown("Monitor and compare machine learning experiments")
        
        try:
            from src.mlflow_utils import MLflowUtils
            # Initialize MLflow utils
            mlflow_utils = MLflowUtils()
            
            # Sidebar for MLflow configuration
            st.sidebar.header("MLflow Settings")
            
            # Get all experiments
            experiments = mlflow_utils.get_experiments()
            
            if not experiments:
                st.warning("No MLflow experiments found. Train some models first!")
                return
            
            experiment_names = [exp.name for exp in experiments]
            selected_experiment = st.sidebar.selectbox("Select Experiment", experiment_names)
            
            # Display experiment information
            st.header(f"Experiment: {selected_experiment}")
            
            # Get runs for selected experiment
            runs = mlflow_utils.get_experiment_runs(selected_experiment)
            
            if not runs:
                st.warning("No runs found for this experiment.")
                return
            
            # Prepare runs data for display - NO METRICS
            run_data = []
            for run in runs:
                data = {
                    'Run ID': run.info.run_id,
                    'Run Name': run.info.run_name,
                    'Status': run.info.status,
                    'Start Time': pd.to_datetime(run.info.start_time, unit='ms'),
                }
                # Only add basic parameters, no metrics
                if run.data.params:
                    # Only keep model_name parameter
                    if 'model_name' in run.data.params:
                        data['Model Name'] = run.data.params['model_name']
                run_data.append(data)
            
            df_runs = pd.DataFrame(run_data)
            
            # Display basic statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Runs", len(df_runs))
            with col2:
                successful_runs = len(df_runs[df_runs['Status'] == 'FINISHED'])
                st.metric("Successful Runs", successful_runs)
            with col3:
                active_runs = len(df_runs[df_runs['Status'] == 'RUNNING'])
                st.metric("Active Runs", active_runs)
            with col4:
                failed_runs = len(df_runs[df_runs['Status'] == 'FAILED'])
                st.metric("Failed Runs", failed_runs)
            
            # Best models identification - Only show names
            st.header("🏆 Selected Models")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Classification Model")
                classification_runs = df_runs[df_runs['Run Name'].str.contains('Classification', na=False)]
                if not classification_runs.empty:
                    # Get model name from parameters or clean run name
                    if 'Model Name' in classification_runs.columns:
                        model_name = classification_runs.iloc[0]['Model Name']
                    else:
                        model_name = classification_runs.iloc[0]['Run Name'].replace('Classification_', '')
                    st.success(f"**{model_name}**")
                else:
                    st.info("No classification models found")
            
            with col2:
                st.subheader("Regression Model")
                regression_runs = df_runs[df_runs['Run Name'].str.contains('Regression', na=False)]
                if not regression_runs.empty:
                    # Get model name from parameters or clean run name
                    if 'Model Name' in regression_runs.columns:
                        model_name = regression_runs.iloc[0]['Model Name']
                    else:
                        model_name = regression_runs.iloc[0]['Run Name'].replace('Regression_', '')
                    st.success(f"**{model_name}**")
                else:
                    st.info("No regression models found")
            
            # Model Categories Overview
            st.header("📊 Model Categories")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Classification Models")
                classification_runs = df_runs[df_runs['Run Name'].str.contains('Classification', na=False)]
                if not classification_runs.empty:
                    for _, run in classification_runs.iterrows():
                        if 'Model Name' in run:
                            model_name = run['Model Name']
                        else:
                            model_name = run['Run Name'].replace('Classification_', '')
                        status_icon = "✅" if run['Status'] == 'FINISHED' else "🔄" if run['Status'] == 'RUNNING' else "❌"
                        st.write(f"{status_icon} {model_name}")
                else:
                    st.info("No classification models trained")
            
            with col2:
                st.subheader("Regression Models")
                regression_runs = df_runs[df_runs['Run Name'].str.contains('Regression', na=False)]
                if not regression_runs.empty:
                    for _, run in regression_runs.iterrows():
                        if 'Model Name' in run:
                            model_name = run['Model Name']
                        else:
                            model_name = run['Run Name'].replace('Regression_', '')
                        status_icon = "✅" if run['Status'] == 'FINISHED' else "🔄" if run['Status'] == 'RUNNING' else "❌"
                        st.write(f"{status_icon} {model_name}")
                else:
                    st.info("No regression models trained")
            
            # Training Progress Visualization
            st.header("📈 Training Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Training Status")
                status_counts = df_runs['Status'].value_counts()
                fig_status = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Training Status",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_status, use_container_width=True)
            
            with col2:
                st.subheader("Model Types")
                # Add model type column
                df_runs['Model Type'] = df_runs['Run Name'].apply(
                    lambda x: 'Classification' if 'Classification' in str(x) else 'Regression'
                )
                type_counts = df_runs['Model Type'].value_counts()
                
                if not type_counts.empty:
                    fig_type = px.bar(
                        x=type_counts.index,
                        y=type_counts.values,
                        title="Model Types",
                        labels={'x': 'Model Type', 'y': 'Count'},
                        color=type_counts.index,
                        color_discrete_sequence=['#1f77b4', '#ff7f0e']
                    )
                    st.plotly_chart(fig_type, use_container_width=True)
                else:
                    st.info("No model type data available")
            
            # Detailed runs table - Only show basic info
            st.header("📋 All Trained Models")
            
            # Create a clean display dataframe
            if 'Model Name' in df_runs.columns:
                display_df = df_runs[['Model Name', 'Run Name', 'Status', 'Start Time']].copy()
            else:
                display_df = df_runs[['Run Name', 'Status', 'Start Time']].copy()
                display_df['Model Name'] = display_df['Run Name'].str.replace('Classification_', '').str.replace('Regression_', '')
            
            # Add model type
            display_df['Type'] = display_df['Run Name'].apply(
                lambda x: 'Classification' if 'Classification' in str(x) else 'Regression'
            )
            
            # Reorder columns
            final_columns = ['Model Name', 'Type', 'Status', 'Start Time']
            display_df = display_df[final_columns]
            
            st.dataframe(display_df, width='stretch')
            
            # Model artifacts and details
            st.header("🔍 Model Details")
            
            if not df_runs.empty:
                # Create a simplified model selection dropdown
                model_options = []
                for _, run in df_runs.iterrows():
                    if 'Model Name' in run:
                        display_name = run['Model Name']
                    else:
                        display_name = run['Run Name'].replace('Classification_', '').replace('Regression_', '')
                    model_options.append((run['Run ID'], display_name))
                
                selected_run_id = st.selectbox(
                    "Select Model for Details", 
                    options=[opt[0] for opt in model_options],
                    format_func=lambda x: next((opt[1] for opt in model_options if opt[0] == x), x)
                )
                
                if selected_run_id:
                    try:
                        from mlflow.tracking import MlflowClient
                        client = MlflowClient()
                        run = client.get_run(selected_run_id)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Model Information")
                            # Get display name
                            if 'model_name' in run.data.params:
                                display_name = run.data.params['model_name']
                            else:
                                display_name = run.info.run_name.replace('Classification_', '').replace('Regression_', '')
                            
                            info_data = {
                                'Model Name': display_name,
                                'Status': run.info.status,
                                'Start Time': pd.to_datetime(run.info.start_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S'),
                                'Type': 'Classification' if 'Classification' in run.info.run_name else 'Regression'
                            }
                            
                            info_df = pd.DataFrame(list(info_data.items()), columns=['Property', 'Value'])
                            st.dataframe(info_df, width='stretch', hide_index=True)
                        
                        with col2:
                            st.subheader("Training Parameters")
                            if run.data.params:
                                # Filter out technical parameters
                                model_params = {k: v for k, v in run.data.params.items() 
                                              if k not in ['model_name', 'model_type', 'run_name'] and not k.startswith('_')}
                                if model_params:
                                    params_df = pd.DataFrame(list(model_params.items()), columns=['Parameter', 'Value'])
                                    st.dataframe(params_df, width='stretch', hide_index=True)
                                else:
                                    st.info("No model parameters recorded.")
                            else:
                                st.info("No parameters recorded for this model.")
                        
                        # Artifacts - NO METRICS
                        st.subheader("Model Artifacts")
                        try:
                            artifacts = client.list_artifacts(selected_run_id)
                            if artifacts:
                                st.success("✅ Model artifacts available:")
                                for artifact in artifacts:
                                    st.write(f"• {artifact.path}")
                            else:
                                st.info("No artifacts found for this model.")
                        except:
                            st.info("Could not load artifacts for this model.")
                    
                    except Exception as e:
                        st.error(f"Error loading model details: {e}")
                        
        except ImportError as e:
            st.error(f"MLflow utilities not available: {e}")
            st.info("Please ensure MLflow is properly configured and models have been trained.")

def main():
    app = EMIPredictApp()
    
    # Updated sidebar navigation with all 5 pages
    st.sidebar.title("🏦 Navigation")
    page = st.sidebar.radio("Go to", [
        "Home", 
        "EDA", 
        "EMI Calculator",
        "Model Training",  # Added Model Training
        "MLflow Dashboard"
    ])
    
    if page == "Home":
        app.home_page()
    elif page == "EDA":
        app.eda_page()
    elif page == "EMI Calculator":
        app.prediction_page()
    elif page == "Model Training":
        # Import and run the Model Training page
        try:
            from models.Model_Training import main as model_training_main
            model_training_main()
        except ImportError:
            st.error("Model Training page not found. Please create pages/Model_Training.py")
            st.info("""
            To enable model training, create a file called `Model_Training.py` in the `pages/` folder.
            This page will allow you to train machine learning models for EMI prediction.
            """)
    elif page == "MLflow Dashboard":
        app.mlflow_dashboard_page()

if __name__ == "__main__":
    main()