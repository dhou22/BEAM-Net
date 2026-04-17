-- ============================================================================
-- BEAM-Net: Database Creation
-- ============================================================================
-- Creates the three databases used by the infrastructure:
--   - airflow       : Airflow metadata (created by postgres image env var)
--   - mlflow        : MLflow experiment backend store
--   - beam_metrics  : Custom analytics schema for BEAM-Net experiments
-- ============================================================================

CREATE DATABASE mlflow;
CREATE DATABASE beam_metrics;