-- ============================================================================
-- BEAM-Net: Role & Permission Setup
-- ============================================================================
-- Grants privileges to the 'beam' service account.
-- In production, you would create separate roles per service:
--   beam_airflow (read/write airflow DB only)
--   beam_mlflow  (read/write mlflow DB only)
--   beam_analyst (read-only on beam_metrics for report generation)
-- For this research project, a single role is acceptable but documented.
-- ============================================================================

GRANT ALL PRIVILEGES ON DATABASE airflow TO beam;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO beam;
GRANT ALL PRIVILEGES ON DATABASE beam_metrics TO beam;