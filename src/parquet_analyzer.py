"""
DuckDB Analytics on Parquet Data
===================================
Query BEAM-Net experiment data stored in MinIO Parquet files using DuckDB.

Why DuckDB over alternatives:
  - Zero-server: no Spark cluster, no separate service to manage
  - Native S3/MinIO support via httpfs extension
  - Reads Parquet with predicate pushdown (only loads needed columns/rows)
  - SQL interface familiar to data engineers and scientists
  - Used at Dataiku, MotherDuck, Criteo for lakehouse analytics

Architecture:
  - Hot data (recent experiments, status): PostgreSQL
  - Cold data (millions of rows): Parquet on MinIO, queried via DuckDB
  - Analysis notebooks load from DuckDB, not Postgres

Usage from report_generator.py:
    analyzer = BeamNetAnalyzer()
    df = analyzer.reliability_diagram_data(experiment_id=42)
    df = analyzer.compare_models(dataset='nmnist')
"""

import os
from typing import Optional
import pandas as pd
import duckdb


class BeamNetAnalyzer:
    """
    Query engine for BEAM-Net Parquet data on MinIO.
    
    Supports both local file:// access (for testing) and s3:// access via
    the DuckDB httpfs extension.
    """
    
    def __init__(self, bucket: str = "beam-net-results"):
        self.bucket = bucket
        self.endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
        self.access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        self.secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        
        self.con = duckdb.connect(":memory:")
        self._configure_s3()
    
    def _configure_s3(self):
        """Set up DuckDB to read from MinIO."""
        # Install and load httpfs extension (enables s3:// URLs)
        self.con.execute("INSTALL httpfs; LOAD httpfs;")
        
        # Configure S3 endpoint for MinIO
        endpoint = self.endpoint.replace("http://", "").replace("https://", "")
        self.con.execute(f"""
            SET s3_endpoint='{endpoint}';
            SET s3_access_key_id='{self.access_key}';
            SET s3_secret_access_key='{self.secret_key}';
            SET s3_url_style='path';
            SET s3_use_ssl=false;
        """)
    
    # ------------------------------------------------------------------------
    # Experiment comparison queries
    # ------------------------------------------------------------------------
    
    def compare_models(self, dataset: str = "nmnist") -> pd.DataFrame:
        """
        Aggregate metrics per model variant — powers Table 1 in the report.
        
        Reads all experiments' predictions from partitioned Parquet and
        computes model-level statistics. Uses DuckDB's Hive partition
        awareness to prune irrelevant files.
        """
        query = f"""
        SELECT
            model,
            COUNT(DISTINCT experiment_id)                AS n_experiments,
            COUNT(*)                                     AS n_predictions,
            AVG(CAST(is_correct AS DOUBLE))             AS accuracy,
            AVG(max_probability)                        AS mean_confidence,
            AVG(epistemic_unc)                          AS mean_epistemic,
            AVG(aleatoric_unc)                          AS mean_aleatoric,
            AVG(CAST(n_iterations AS DOUBLE))           AS mean_iterations
        FROM read_parquet('s3://{self.bucket}/predictions/dataset={dataset}/**/*.parquet',
                          hive_partitioning=1)
        GROUP BY model
        ORDER BY accuracy DESC;
        """
        return self.con.execute(query).fetchdf()
    
    def reliability_diagram_data(
        self, experiment_id: int, n_bins: int = 15
    ) -> pd.DataFrame:
        """
        Compute reliability diagram bins for calibration analysis.
        
        Leverages DuckDB's WIDTH_BUCKET for efficient binning without
        loading data into pandas first.
        """
        query = f"""
        WITH binned AS (
            SELECT
                WIDTH_BUCKET(max_probability, 0, 1, {n_bins}) AS bin_number,
                max_probability,
                CAST(is_correct AS DOUBLE) AS correct
            FROM read_parquet('s3://{self.bucket}/predictions/**/*.parquet',
                              hive_partitioning=1)
            WHERE experiment_id = {experiment_id}
        )
        SELECT
            bin_number,
            (bin_number - 0.5) / {n_bins}           AS bin_center,
            COUNT(*)                                AS n_samples,
            AVG(max_probability)                    AS mean_confidence,
            AVG(correct)                            AS accuracy,
            ABS(AVG(max_probability) - AVG(correct)) AS calibration_gap
        FROM binned
        WHERE bin_number > 0
        GROUP BY bin_number
        ORDER BY bin_number;
        """
        return self.con.execute(query).fetchdf()
    
    # ------------------------------------------------------------------------
    # Energy analytics
    # ------------------------------------------------------------------------
    
    def energy_scaling_data(self) -> pd.DataFrame:
        """
        Aggregate energy measurements across experiments for scaling plot.
        """
        query = f"""
        SELECT
            platform,
            experiment_id,
            AVG(energy_nj)              AS mean_energy_nj,
            AVG(sparsity)               AS mean_sparsity,
            SUM(spike_count)            AS total_spikes,
            COUNT(*)                    AS n_batches
        FROM read_parquet('s3://{self.bucket}/energy_timeseries/**/*.parquet',
                          hive_partitioning=1)
        GROUP BY platform, experiment_id
        ORDER BY platform, experiment_id;
        """
        return self.con.execute(query).fetchdf()
    
    def energy_evolution(self, experiment_id: int) -> pd.DataFrame:
        """
        Per-epoch energy progression — shows how sparsity growth during
        training reduces per-batch energy over time.
        """
        query = f"""
        SELECT
            epoch,
            AVG(sparsity)       AS mean_sparsity,
            AVG(energy_nj)      AS mean_energy_nj,
            MIN(energy_nj)      AS min_energy_nj,
            MAX(energy_nj)      AS max_energy_nj,
            COUNT(*)            AS n_batches
        FROM read_parquet('s3://{self.bucket}/energy_timeseries/**/*.parquet',
                          hive_partitioning=1)
        WHERE experiment_id = {experiment_id}
        GROUP BY epoch
        ORDER BY epoch;
        """
        return self.con.execute(query).fetchdf()
    
    # ------------------------------------------------------------------------
    # Uncertainty analytics
    # ------------------------------------------------------------------------
    
    def uncertainty_vs_correctness(self, experiment_id: int) -> pd.DataFrame:
        """
        Validate that epistemic uncertainty correlates with errors.
        
        This is the scientific test of the Dirichlet attention claim:
        if the uncertainty quantification is meaningful, incorrectly-
        classified samples should have HIGHER epistemic uncertainty than
        correctly-classified ones.
        """
        query = f"""
        SELECT
            is_correct,
            COUNT(*)                AS n_samples,
            AVG(epistemic_unc)      AS mean_epistemic,
            AVG(aleatoric_unc)      AS mean_aleatoric,
            AVG(max_probability)    AS mean_confidence
        FROM read_parquet('s3://{self.bucket}/predictions/**/*.parquet',
                          hive_partitioning=1)
        WHERE experiment_id = {experiment_id}
        GROUP BY is_correct;
        """
        return self.con.execute(query).fetchdf()
    
    # ------------------------------------------------------------------------
    # Spike trace analytics (highest-volume data)
    # ------------------------------------------------------------------------
    
    def neuron_selectivity(self, experiment_id: int, top_k: int = 20) -> pd.DataFrame:
        """
        Identify most selective neurons across all spike traces.
        
        High selectivity = few, strong responses → good representation learning.
        Low selectivity = many weak responses → grandmother-cell-like behavior
        not achieved.
        
        Only feasible with Parquet: summing 10M+ spike rows.
        """
        query = f"""
        SELECT
            neuron_id,
            COUNT(*)                    AS total_spikes,
            COUNT(DISTINCT sample_idx)  AS distinct_samples,
            CAST(COUNT(*) AS DOUBLE) / COUNT(DISTINCT sample_idx) AS spikes_per_sample
        FROM read_parquet('s3://{self.bucket}/spike_traces/**/*.parquet',
                          hive_partitioning=1)
        WHERE experiment_id = {experiment_id}
        GROUP BY neuron_id
        ORDER BY spikes_per_sample DESC
        LIMIT {top_k};
        """
        return self.con.execute(query).fetchdf()
    
    def close(self):
        self.con.close()