import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys

sys.path.append('src')

# Add data type fixing function at the top
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

try:
    from src.data_preprocessing import DataPreprocessor
    from src.config import Config
except ImportError as e:
    st.error(f"Import error: {e}")
    # Fallback classes if imports fail
    class DataPreprocessor:
        pass
    class Config:
        RAW_DATA_PATH = "data/raw/emi_prediction_dataset.csv"

def create_simple_histogram(df, column, title):
    """Create histogram without deprecated parameters"""
    fig = px.histogram(df, x=column, nbins=30)
    fig.update_layout(
        title=title,
        xaxis_title=column,
        yaxis_title="Count",
        showlegend=False
    )
    return fig

def create_simple_bar_chart(labels, values, title, xaxis_title):
    """Create bar chart without deprecated parameters"""
    fig = go.Figure(data=[go.Bar(x=labels, y=values)])
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Count",
        showlegend=False
    )
    return fig

def create_simple_correlation_heatmap(corr_matrix, title):
    """Create correlation heatmap without deprecated parameters"""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r'
    ))
    fig.update_layout(
        title=title,
        xaxis_tickangle=-45
    )
    return fig

def create_simple_box_plot(df, x_col, y_col, title):
    """Create box plot without deprecated parameters"""
    if x_col:  # With categorical variable
        categories = df[x_col].unique()
        data = []
        for category in categories:
            data.append(go.Box(
                y=df[df[x_col] == category][y_col],
                name=str(category)
            ))
        fig = go.Figure(data=data)
    else:  # Single box plot
        fig = go.Figure(data=[go.Box(y=df[y_col])])
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col if x_col else "",
        yaxis_title=y_col,
        showlegend=False
    )
    return fig

def main():
    st.set_page_config(page_title="EDA - EMI Predict AI", layout="wide")
    
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Comprehensive analysis of the EMI prediction dataset")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data with proper error handling
    dataset_path = getattr(Config, 'RAW_DATA_PATH', 'data/raw/emi_prediction_dataset.csv')
    
    if not os.path.exists(dataset_path):
        st.error(f"Dataset not found at {dataset_path}")
        st.info("Please ensure 'emi_prediction_dataset.csv' is in the data/raw/ directory")
        return
    
    try:
        # FIXED: Load CSV with proper type handling
        df = pd.read_csv(dataset_path, low_memory=False)
        
        # Fix the first column if it has mixed types
        if len(df.columns) > 0:
            first_col = df.columns[0]
            df[first_col] = df[first_col].astype(str)
            
        st.success(f"Dataset loaded successfully: {df.shape}")
        
        # Basic information
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Number of Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Data Types", f"{len(df.dtypes.unique())}")
        
        # Dataset overview
        st.subheader("📋 Dataset Overview")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Data Sample", "Data Types", "Missing Values", "Descriptive Stats"])
        
        with tab1:
            st.write("First 10 records:")
            display_df1 = fix_dataframe_for_display(df.head(10))
            st.dataframe(display_df1, width='stretch')
        
        with tab2:
            dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
            dtype_df['Non-Null Count'] = df.count()
            display_df2 = fix_dataframe_for_display(dtype_df)
            st.dataframe(display_df2, width='stretch')
        
        with tab3:
            missing_df = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
            missing_df['Percentage'] = (missing_df['Missing Values'] / len(df)) * 100
            missing_df = missing_df[missing_df['Missing Values'] > 0]
            if len(missing_df) > 0:
                display_df3 = fix_dataframe_for_display(missing_df)
                st.dataframe(display_df3, width='stretch')
            else:
                st.success("No missing values found in the dataset!")
        
        with tab4:
            display_df4 = fix_dataframe_for_display(df.describe())
            st.dataframe(display_df4, width='stretch')
        
        # Visualizations
        st.subheader("📈 Data Visualizations")
        
        # Numerical features distribution
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_num_col = st.selectbox("Select numerical feature for distribution:", numerical_cols)
                if selected_num_col:
                    # FIXED: Use simple histogram function
                    fig = create_simple_histogram(df, selected_num_col, f'Distribution of {selected_num_col}')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Correlation heatmap for first 10 numerical features
                if len(numerical_cols) >= 2:
                    corr_cols = numerical_cols[:10]  # First 10 columns for performance
                    corr_matrix = df[corr_cols].corr()
                    
                    # FIXED: Use simple correlation heatmap function
                    fig = create_simple_correlation_heatmap(corr_matrix, 'Correlation Heatmap')
                    st.plotly_chart(fig, use_container_width=True)
        
        # Categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_cat_col = st.selectbox("Select categorical feature:", categorical_cols)
                if selected_cat_col:
                    value_counts = df[selected_cat_col].value_counts().head(15)  # Limit to top 15
                    
                    # FIXED: Use simple bar chart function
                    fig = create_simple_bar_chart(
                        value_counts.index, 
                        value_counts.values, 
                        f'Distribution of {selected_cat_col}',
                        selected_cat_col
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot for numerical vs categorical
                if len(numerical_cols) > 0 and len(categorical_cols) > 0:
                    num_col = st.selectbox("Select numerical feature:", numerical_cols, key='box_num')
                    cat_col = st.selectbox("Select categorical feature:", categorical_cols, key='box_cat')
                    
                    if num_col and cat_col:
                        # FIXED: Use simple box plot function
                        fig = create_simple_box_plot(df, cat_col, num_col, f'{num_col} by {cat_col}')
                        st.plotly_chart(fig, use_container_width=True)
        
        # Outlier detection
        st.subheader("🔍 Outlier Analysis")
        
        if len(numerical_cols) > 0:
            outlier_col = st.selectbox("Select feature for outlier detection:", numerical_cols, key='outlier')
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
                    st.metric("Lower Bound", f"{lower_bound:.2f}")
                    st.metric("Upper Bound", f"{upper_bound:.2f}")
                
                with col2:
                    # FIXED: Use simple box plot function
                    fig = create_simple_box_plot(df, None, outlier_col, f'Box Plot of {outlier_col}')
                    st.plotly_chart(fig, use_container_width=True)
                
                if len(outliers) > 0:
                    with st.expander("View Outlier Data"):
                        display_df5 = fix_dataframe_for_display(outliers)
                        st.dataframe(display_df5, width='stretch')
        
        # Data Quality Report
        st.subheader("📊 Data Quality Report")
        
        quality_data = []
        for col in df.columns:
            quality_data.append({
                'Column': col,
                'Data Type': str(df[col].dtype),
                'Total Values': len(df[col]),
                'Missing Values': df[col].isnull().sum(),
                'Missing %': (df[col].isnull().sum() / len(df[col])) * 100,
                'Unique Values': df[col].nunique(),
                'Unique %': (df[col].nunique() / len(df[col])) * 100
            })
        
        quality_df = pd.DataFrame(quality_data)
        display_df6 = fix_dataframe_for_display(quality_df)
        st.dataframe(display_df6, width='stretch')
        
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.info("This might be due to data type issues in the CSV file. Try checking your dataset format.")

if __name__ == "__main__":
    main()