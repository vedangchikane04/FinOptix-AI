import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .config import Config

class MLflowUtils:
    def __init__(self):
        self.client = MlflowClient()
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    
    def get_experiments(self):
        """Get all experiments"""
        return self.client.search_experiments()
    
    def get_experiment_runs(self, experiment_name):
        """Get all runs for an experiment - NO METRICS"""
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment:
            return self.client.search_runs(experiment.experiment_id)
        return []
    
    def get_best_run(self, experiment_name):
        """Get the best run based on model type - NO SCORES"""
        runs = self.get_experiment_runs(experiment_name)
        if not runs:
            return None
        
        # Just return the first run of each type
        classification_runs = [r for r in runs if 'Classification' in r.info.run_name]
        regression_runs = [r for r in runs if 'Regression' in r.info.run_name]
        
        best_runs = {}
        if classification_runs:
            best_runs['classification'] = classification_runs[0]
        if regression_runs:
            best_runs['regression'] = regression_runs[0]
        
        return best_runs
    
    def create_training_dashboard(self, experiment_name):
        """Create a training dashboard without scores"""
        runs = self.get_experiment_runs(experiment_name)
        if not runs:
            return None
        
        # Prepare data without metrics
        run_data = []
        for run in runs:
            data = {
                'run_name': run.info.run_name,
                'status': run.info.status,
                'start_time': pd.to_datetime(run.info.start_time, unit='ms'),
            }
            # Only add parameters, no metrics
            if run.data.params:
                data.update(run.data.params)
            run_data.append(data)
        
        df_runs = pd.DataFrame(run_data)
        
        # Separate classification and regression runs
        classification_runs = df_runs[df_runs['run_name'].str.contains('Classification')]
        regression_runs = df_runs[df_runs['run_name'].str.contains('Regression')]
        
        return df_runs, classification_runs, regression_runs
    
    def plot_model_distribution(self, classification_runs, regression_runs):
        """Create model distribution plots without scores"""
        # Model type distribution
        fig_type = go.Figure()
        
        if not classification_runs.empty:
            fig_type.add_trace(go.Bar(
                x=['Classification'],
                y=[len(classification_runs)],
                name='Classification',
                marker_color='#1f77b4'
            ))
        
        if not regression_runs.empty:
            fig_type.add_trace(go.Bar(
                x=['Regression'],
                y=[len(regression_runs)],
                name='Regression',
                marker_color='#ff7f0e'
            ))
        
        fig_type.update_layout(
            title="Model Type Distribution",
            xaxis_title="Model Type",
            yaxis_title="Number of Models",
            showlegend=False
        )
        
        return fig_type