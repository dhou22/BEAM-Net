-- ============================================================================
-- BEAM-Net: Schema Migration V002
-- ============================================================================
-- Purpose: Performance optimizations and partitioning for large experiment runs.
-- Apply once you have >100 experiments or >1M prediction rows.
-- ============================================================================

\c beam_metrics;

-- Partial index for active (running) experiments — common dashboard query
CREATE INDEX IF NOT EXISTS idx_experiments_running
    ON experiments(started_at DESC)
    WHERE status = 'running';

-- Composite index for calibration analysis queries
CREATE INDEX IF NOT EXISTS idx_test_pred_calibration
    ON test_predictions(experiment_id, max_probability, is_correct);

-- Add comment documentation for CIFRE-quality schema inspection
COMMENT ON TABLE experiments IS
    'Master experiment registry. One row per training run. Joins to MLflow via mlflow_run_id.';

COMMENT ON TABLE epoch_metrics IS
    'Per-epoch training dynamics. Supports training curve reconstruction and monotonic descent validation (Theorem 4.1).';

COMMENT ON TABLE test_predictions IS
    'Per-sample test-set predictions with uncertainty. Enables post-hoc calibration studies and failure-mode analysis.';

COMMENT ON TABLE energy_measurements IS
    'Theoretical energy accounting (Eq. 25). Loihi 2 projections based on Davies et al., 2021 (23.6 pJ/spike).';

COMMENT ON TABLE sparsity_by_layer IS
    'Layer-wise activity sparsity. Distinguishes encoding vs representation layer (W-TCRL §3.1.2).';