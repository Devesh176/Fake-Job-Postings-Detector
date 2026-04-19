import mlflow
import json
import os

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'fake-job-detector-exp')
def register_model(min_f1=0.85, min_auc=0.90):
    with open('data/processed/eval_metrics.json') as f:
        metrics = json.load(f)

    if metrics['f1_fraud'] >= min_f1 and metrics['roc_auc'] >= min_auc:
        # Find the latest run in the experiment
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=['start_time DESC'],
            max_results=1
        )
        run_id = runs[0].info.run_id

        model_uri = f'runs:/{run_id}/model'
        result = mlflow.register_model(model_uri, 'FakeJobDetector')

        client.transition_model_version_stage(
            name='FakeJobDetector',
            version=result.version,
            stage='Production',
            archive_existing_versions=True
        )
        print(f"Model v{result.version} promoted to Production. Run ID: {run_id}")
    else:
        print(f"Quality gates NOT met. F1={metrics['f1_fraud']:.3f}, AUC={metrics['roc_auc']:.3f}")
        raise ValueError("Model did not meet quality gates — not registering.")

if __name__ == '__main__':
    register_model()