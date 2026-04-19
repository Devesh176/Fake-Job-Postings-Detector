import mlflow
import pandas as pd
import os

# Config
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "fake-job-detector-exp")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "mlflow_runs.csv")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def export_runs():
    # Get experiment
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found")

    # Fetch runs
    runs_df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        output_format="pandas"
    )

    if runs_df.empty:
        print("No runs found.")
        return

    # Optional: clean column names (remove prefixes)
    runs_df.columns = [
        col.replace("metrics.", "")
           .replace("params.", "")
           .replace("tags.", "")
        for col in runs_df.columns
    ]

    # Save to CSV
    runs_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Exported {len(runs_df)} runs to {OUTPUT_FILE}")


if __name__ == "__main__":
    export_runs()