import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import mlflow
from mlflow.tracking import MlflowClient
import sys
import os

sys.path.append('src')
from src.mlflow_utils import MLflowUtils
from src.config import Config

def main():
    st.set_page_config(page_title="MLflow Dashboard - EMI Predict AI", layout="wide")
    
    st.title("📈 MLflow Experiment Tracking")
    st.markdown("Monitor and compare machine learning experiments")
    
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
    
    # Prepare runs data for display - NO METRICS AT ALL
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
    
    st.dataframe(display_df, use_container_width=True)
    
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
                client = MlflowClient()
                run = client.get_run(selected_run_id)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Model Information")
                    # Get display name
                    if 'Model Name' in run.data.params:
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
                    st.dataframe(info_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.subheader("Training Parameters")
                    if run.data.params:
                        # Filter out technical parameters
                        model_params = {k: v for k, v in run.data.params.items() 
                                      if k not in ['model_name', 'model_type', 'run_name'] and not k.startswith('_')}
                        if model_params:
                            params_df = pd.DataFrame(list(model_params.items()), columns=['Parameter', 'Value'])
                            st.dataframe(params_df, use_container_width=True, hide_index=True)
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

if __name__ == "__main__":
    main()