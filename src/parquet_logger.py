"""
Parquet Writer for BEAM-Net Analytics
========================================
Writes high-volume experimental data to MinIO as partitioned Parquet files.

Architecture rationale:
  - PostgreSQL: hot operational data (experiment status, aggregated summaries)
  - Parquet on MinIO: cold analytical data (per-sample predictions, spike traces)
  - DuckDB: queries Parquet directly without loading into memory

This follows the "data lakehouse" pattern — standard at Criteo, Dataiku,
Datadog, and other French tech companies relevant to CIFRE partnerships.

Why Parquet for BEAM-Net:
  - Columnar compression: 5-10x smaller than CSV/JSON for our schema
  - Per-column reads: analyzing just `epistemic_unc` doesn't load other columns
  - Snappy/ZSTD compression: optimized for numerical scientific data
  - Immutable snapshots: each experiment run = versioned dataset
  - Interoperable: readable by pandas, polars, duckdb, spark, arrow

Partition strategy:
  s3://beam-net-results/
    predictions/
      dataset=nmnist/model=beam_net/experiment_id=42/predictions.parquet
    energy_timeseries/
      dataset=nmnist/platform=loihi2/experiment_id=42/energy.parquet
    spike_traces/
      experiment_id=42/batch=0/trace.parquet

Partition keys (dataset, model, experiment_id) are Hive-style — readable
by any SQL-on-object-store tool without a catalog service.
"""

import os
import io
from typing import Optional, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import boto3
from botocore.client import Config


class ParquetMinIOWriter:
    """
    Writes partitioned Parquet files to MinIO (S3-compatible).
    
    Parameters
    ----------
    endpoint_url : str
        MinIO endpoint, e.g., 'http://minio:9000'.
    access_key : str
        MinIO access key.
    secret_key : str
        MinIO secret key.
    bucket : str
        Target bucket name.
    compression : str
        Parquet compression codec: 'snappy' (fast), 'zstd' (smaller), 'gzip'.
    """
    
    def __init__(
        self,
        endpoint_url: str = None,
        access_key: str = None,
        secret_key: str = None,
        bucket: str = "beam-net-results",
        compression: str = "zstd",
    ):
        # Read from environment if not provided (Docker-friendly)
        self.endpoint_url = endpoint_url or os.environ.get("MLFLOW_S3_ENDPOINT_URL")
        self.access_key = access_key or os.environ.get("AWS_ACCESS_KEY_ID")
        self.secret_key = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.bucket = bucket
        self.compression = compression
        
        # S3 client configured for MinIO (path-style URLs, no region)
        self.s3 = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
    
    def _build_key(self, dataset_type: str, partitions: Dict[str, Any], filename: str) -> str:
        """
        Build Hive-style partitioned S3 key.
        
        Example:
          dataset_type='predictions', 
          partitions={'dataset': 'nmnist', 'model': 'beam_net', 'experiment_id': 42}
          filename='predictions.parquet'
        →
          'predictions/dataset=nmnist/model=beam_net/experiment_id=42/predictions.parquet'
        """
        partition_path = "/".join(f"{k}={v}" for k, v in partitions.items())
        return f"{dataset_type}/{partition_path}/{filename}"
    
    def write_dataframe(
        self,
        df: pd.DataFrame,
        dataset_type: str,
        partitions: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> str:
        """
        Write a DataFrame to MinIO as Parquet.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to write.
        dataset_type : str
            Top-level folder: 'predictions', 'energy_timeseries', 'spike_traces'.
        partitions : dict
            Hive-style partition key-values.
        filename : str, optional
            File name. Auto-generated with timestamp if not provided.
        
        Returns
        -------
        s3_uri : str
            Full S3 URI of written file (s3://bucket/key).
        """
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            filename = f"{dataset_type}_{ts}.parquet"
        
        key = self._build_key(dataset_type, partitions, filename)
        
        # Convert DataFrame → Arrow Table → Parquet bytes in memory
        table = pa.Table.from_pandas(df, preserve_index=False)
        buffer = io.BytesIO()
        pq.write_table(
            table, buffer,
            compression=self.compression,
            use_dictionary=True,        # Dict encoding for categorical columns
            write_statistics=True,       # Enable predicate pushdown
        )
        buffer.seek(0)
        
        # Upload to MinIO
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer.getvalue(),
            ContentType="application/vnd.apache.parquet",
        )
        
        s3_uri = f"s3://{self.bucket}/{key}"
        print(f"[Parquet] Wrote {len(df)} rows → {s3_uri} "
              f"({buffer.tell() / 1024:.1f} KB, compression={self.compression})")
        return s3_uri


# ---- Specialized writers for BEAM-Net data types ----

class BeamNetParquetLogger:
    """
    High-level API for logging BEAM-Net experimental data to Parquet.
    
    Wraps ParquetMinIOWriter with domain-specific methods that produce
    correctly-typed DataFrames following the BEAM-Net data schema.
    """
    
    def __init__(self, experiment_id: int, dataset: str, model_variant: str):
        self.writer = ParquetMinIOWriter()
        self.experiment_id = experiment_id
        self.dataset = dataset
        self.model_variant = model_variant
    
    def log_test_predictions(
        self,
        sample_ids: np.ndarray,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        max_probs: np.ndarray,
        epistemic_unc: np.ndarray,
        aleatoric_unc: np.ndarray,
        n_iterations: Optional[np.ndarray] = None,
    ) -> str:
        """
        Log per-sample test predictions with uncertainty.
        
        This is the data that would overwhelm PostgreSQL at scale:
        10,000 test samples × 50 experiments = 500k rows, manageable in Postgres
        but 50 experiments × 10k samples × multiple seeds × long training = 10M+.
        Parquet handles this trivially.
        """
        df = pd.DataFrame({
            "sample_id": sample_ids.astype(np.int32),
            "true_label": true_labels.astype(np.int16),
            "predicted_label": predicted_labels.astype(np.int16),
            "max_probability": max_probs.astype(np.float32),
            "epistemic_unc": epistemic_unc.astype(np.float32),
            "aleatoric_unc": aleatoric_unc.astype(np.float32),
            "n_iterations": (n_iterations if n_iterations is not None 
                            else np.zeros(len(sample_ids))).astype(np.int8),
            "is_correct": (true_labels == predicted_labels).astype(bool),
        })
        
        return self.writer.write_dataframe(
            df,
            dataset_type="predictions",
            partitions={
                "dataset": self.dataset,
                "model": self.model_variant,
                "experiment_id": self.experiment_id,
            },
        )
    
    def log_energy_timeseries(
        self,
        epoch: int,
        batch_indices: np.ndarray,
        sparsity: np.ndarray,
        spike_counts: np.ndarray,
        energy_nj: np.ndarray,
        platform: str = "loihi2",
    ) -> str:
        """
        Log per-batch energy measurements during training.
        
        High-frequency data: 60,000 samples / 64 batch_size = ~940 batches/epoch.
        50 epochs × 940 = 47,000 rows per experiment. Per 50 experiments: 2.35M rows.
        Parquet with ZSTD compression: ~20 MB total vs ~200 MB in PostgreSQL.
        """
        df = pd.DataFrame({
            "epoch": np.full(len(batch_indices), epoch, dtype=np.int16),
            "batch_idx": batch_indices.astype(np.int32),
            "sparsity": sparsity.astype(np.float32),
            "spike_count": spike_counts.astype(np.int64),
            "energy_nj": energy_nj.astype(np.float32),
            "timestamp": pd.Timestamp.utcnow(),
        })
        
        return self.writer.write_dataframe(
            df,
            dataset_type="energy_timeseries",
            partitions={
                "dataset": self.dataset,
                "platform": platform,
                "experiment_id": self.experiment_id,
                "epoch": epoch,
            },
        )
    
    def log_spike_traces(
        self,
        batch_idx: int,
        spike_tensor: np.ndarray,
        neuron_ids: Optional[np.ndarray] = None,
    ) -> str:
        """
        Log spike raster for post-hoc neural dynamics analysis.
        
        Spike traces are the highest-volume data — sparse binary tensors over
        time. Storing only non-zero spikes in long format gives massive savings.
        
        Parameters
        ----------
        spike_tensor : np.ndarray, shape (batch, n_steps, N)
            Full spike trains from BayesianLIFLayer.
        """
        # Extract only non-zero spikes (sparse representation)
        # This reduces 99% of rows when sparsity is ~1% (typical BEAM-Net regime)
        batch_ids, time_ids, neuron_ids_arr = np.where(spike_tensor > 0)
        
        df = pd.DataFrame({
            "sample_idx": batch_ids.astype(np.int32),
            "time_step": time_ids.astype(np.int16),
            "neuron_id": neuron_ids_arr.astype(np.int16),
        })
        
        return self.writer.write_dataframe(
            df,
            dataset_type="spike_traces",
            partitions={
                "experiment_id": self.experiment_id,
                "batch": batch_idx,
            },
        )