"""
BEAM-Net Pipeline DAG (Apache Airflow)
=========================================
Orchestrates the complete experimental pipeline:

  download_data → train_model → evaluate_model → generate_report

Each task is idempotent and logs to MLflow. The DAG can be triggered
manually from the Airflow UI (http://localhost:8080) or scheduled.

Task dependencies enforce the scientific workflow:
  1. Data must be downloaded before training
  2. Model must be trained before evaluation
  3. Evaluation results must exist before report generation

Designed for reproducibility: all parameters come from /opt/airflow/configs/experiment.yaml,
all results are versioned via MLflow, all artifacts stored in MinIO.
"""

import sys
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add src to path so imports work inside Airflow workers
sys.path.insert(0, "/opt/airflow")
sys.path.insert(0, "/workspace")


# ---- Default DAG arguments ----
default_args = {
    "owner": "beam-net",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ---- Task functions ----

def task_download_data(**kwargs):
    """
    Download and prepare datasets.
    
    Uses tonic library to fetch N-MNIST or DVSGesture.
    Falls back to torchvision MNIST if tonic unavailable.
    Data is cached in ./data/ for subsequent runs.
    """
    from src.config import load_config
    from src.data_loader import get_dataloaders
    
    cfg = load_config("/opt/airflow/configs/experiment.yaml")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    
    print(f"[DAG] Data ready: {len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test")


def task_train_model(**kwargs):
    """
    Train BEAM-Net with full MLflow tracking.
    
    Runs src/train.py which:
      - Builds the 4-component architecture
      - Trains with cosine LR schedule
      - Logs all metrics to MLflow
      - Saves best model to results/best_model.pt
      - Saves training history to results/training_history.json
    """
    from src.train import train
    
    history, test_results = train("configs/experiment.yaml")
    
    print(f"[DAG] Training complete. Test accuracy: {test_results['test_accuracy']:.4f}")
    return test_results


def task_evaluate_model(**kwargs):
    """
    Run full evaluation suite: BEAM-Net + baselines.
    
    Runs src/evaluate.py which:
      - Loads best BEAM-Net model
      - Trains and evaluates ANN-MLP baseline
      - Trains and evaluates Rate-coded SNN baseline
      - Computes all metrics (accuracy, ECE, sparsity, energy)
      - Saves comparison to results/evaluation_results.json
    """
    from src.evaluate import run_evaluation
    
    results = run_evaluation("/opt/airflow/configs/experiment.yaml")
    
    for model, metrics in results.items():
        print(f"[DAG] {model}: acc={metrics.get('accuracy', 0):.4f}, "
              f"ece={metrics.get('ece', 0):.4f}")
    
    return results


def task_generate_report(**kwargs):
    """
    Generate scientific PDF report with all results.
    
    Runs src/report_generator.py which:
      - Loads training history and evaluation results
      - Generates 5 publication-quality plots
      - Compiles 6-section PDF report
      - Saves to results/BEAM_Net_Report.pdf
    """
    from src.report_generator import generate_report
    
    report_path = generate_report("/opt/airflow/configs/experiment.yaml")
    print(f"[DAG] Report generated: {report_path}")
    
    # Log report as MLflow artifact
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("beam_net_v1")
        with mlflow.start_run(run_name="report_generation"):
            mlflow.log_artifact(report_path)
            # Also log individual plots
            results_dir = os.path.dirname(report_path)
            for f in os.listdir(results_dir):
                if f.endswith(".png"):
                    mlflow.log_artifact(os.path.join(results_dir, f))
    except Exception as e:
        print(f"[DAG] MLflow artifact logging skipped: {e}")


# ---- DAG Definition ----

with DAG(
    dag_id="beam_net_pipeline",
    default_args=default_args,
    description=(
        "BEAM-Net full experimental pipeline: "
        "data download → model training → evaluation → report generation"
    ),
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["beam-net", "snn", "bayesian", "neuromorphic"],
) as dag:
    
    t_download = PythonOperator(
        task_id="download_data",
        python_callable=task_download_data,
        doc="Download N-MNIST or fallback MNIST dataset via tonic/torchvision",
    )
    
    t_train = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
        doc="Train BEAM-Net with MLflow tracking, save best model",
    )
    
    t_evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=task_evaluate_model,
        doc="Evaluate BEAM-Net + baselines, compute all metrics",
    )
    
    t_report = PythonOperator(
        task_id="generate_report",
        python_callable=task_generate_report,
        doc="Generate scientific PDF report with plots and tables",
    )
    
    # Pipeline: data → train → evaluate → report
    t_download >> t_train >> t_evaluate >> t_report