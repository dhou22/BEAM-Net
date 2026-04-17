-- ============================================================================
-- BEAM-Net: Experiment Schema (Migration V001)
-- ============================================================================
-- Custom schema for BEAM-Net-specific metrics not captured by MLflow:
--   - Per-sample uncertainty tracking (epistemic/aleatoric decomposition)
--   - Bidirectional inference iteration counts (Theorem 3.6 validation)
--   - Sparsity evolution across layers
--   - Energy accounting with Loihi 2 projections
--
-- MLflow captures: params, metrics, artifacts at the *run* level.
-- This schema captures: fine-grained per-sample, per-iteration, per-batch data
-- that the report generator queries for detailed analysis.
-- ============================================================================

\c beam_metrics;

-- ----------------------------------------------------------------------------
-- Table: experiments
-- One row per experimental run. Foreign keys out to all detail tables.
-- Mirrors MLflow run_id for cross-system joins.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiments (
    id                  SERIAL PRIMARY KEY,
    mlflow_run_id       VARCHAR(64) UNIQUE NOT NULL,
    name                VARCHAR(128) NOT NULL,
    dataset             VARCHAR(64) NOT NULL,
    model_variant       VARCHAR(32) NOT NULL,  -- 'beam_net', 'ann_mlp', 'rate_snn'
    n_patterns          INT NOT NULL,           -- N from Eq. 12
    seed                INT NOT NULL,
    started_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at        TIMESTAMP,
    status              VARCHAR(16) NOT NULL DEFAULT 'running',  -- running | complete | failed
    config_yaml         TEXT,                   -- Full config snapshot for reproducibility
    git_commit          VARCHAR(40),            -- Code version pointer
    CONSTRAINT valid_status CHECK (status IN ('running', 'complete', 'failed'))
);

CREATE INDEX idx_experiments_dataset ON experiments(dataset);
CREATE INDEX idx_experiments_variant ON experiments(model_variant);
CREATE INDEX idx_experiments_started ON experiments(started_at DESC);

-- ----------------------------------------------------------------------------
-- Table: epoch_metrics
-- Per-epoch training metrics. Granular than MLflow's step logging.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS epoch_metrics (
    id                  BIGSERIAL PRIMARY KEY,
    experiment_id       INT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    epoch               INT NOT NULL,
    train_loss          REAL NOT NULL,
    val_loss            REAL,
    val_accuracy        REAL,
    val_ece             REAL,                   -- Expected Calibration Error
    sparsity            REAL,                   -- Fraction active neurons (cf. W-TCRL Eq. 7)
    epistemic_unc       REAL,                   -- Eq. 14
    aleatoric_unc       REAL,                   -- Eq. 15
    mean_iterations     REAL,                   -- Bidirectional loop avg (Thm. 3.6)
    learning_rate       REAL,
    epoch_duration_s    REAL,
    logged_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (experiment_id, epoch)
);

CREATE INDEX idx_epoch_metrics_exp ON epoch_metrics(experiment_id, epoch);

-- ----------------------------------------------------------------------------
-- Table: test_predictions
-- Per-sample test predictions with uncertainty. Enables post-hoc
-- calibration analysis, reliability diagrams, and failure mode studies.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS test_predictions (
    id                  BIGSERIAL PRIMARY KEY,
    experiment_id       INT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    sample_id           INT NOT NULL,           -- Index in test set
    true_label          INT NOT NULL,
    predicted_label     INT NOT NULL,
    max_probability     REAL NOT NULL,          -- Confidence
    epistemic_unc       REAL,
    aleatoric_unc       REAL,
    n_iterations_used   INT,                    -- Early-exit count (Thm. 3.6)
    is_correct          BOOLEAN GENERATED ALWAYS AS (true_label = predicted_label) STORED
);

CREATE INDEX idx_test_pred_exp ON test_predictions(experiment_id);
CREATE INDEX idx_test_pred_correct ON test_predictions(experiment_id, is_correct);

-- ----------------------------------------------------------------------------
-- Table: energy_measurements
-- Theoretical energy accounting per inference (Eq. 25).
-- Supports the 22-30x reduction claim with traceable component breakdown.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS energy_measurements (
    id                  BIGSERIAL PRIMARY KEY,
    experiment_id       INT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    platform            VARCHAR(32) NOT NULL,   -- 'loihi2_projected' | 'gpu_a100' | 'biological'
    n_patterns          INT NOT NULL,
    d_input             INT NOT NULL,
    sparsity_factor     REAL NOT NULL,          -- rho in Eq. 25
    n_iterations        INT NOT NULL,           -- L in Eq. 25
    e_encoding_pj       REAL,                   -- Component breakdown
    e_coincidence_pj    REAL,
    e_selection_pj      REAL,
    total_nj            REAL NOT NULL,
    reduction_vs_gpu    REAL,                   -- Derived metric
    logged_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_energy_exp ON energy_measurements(experiment_id);
CREATE INDEX idx_energy_platform ON energy_measurements(platform);

-- ----------------------------------------------------------------------------
-- Table: sparsity_by_layer
-- Layer-wise sparsity tracking. Distinguishes encoding layer sparsity
-- from representation layer sparsity (cf. W-TCRL §3.1.2).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sparsity_by_layer (
    id                  BIGSERIAL PRIMARY KEY,
    experiment_id       INT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    epoch               INT NOT NULL,
    layer_name          VARCHAR(32) NOT NULL,   -- 'encoding' | 'representation'
    spike_count_mean    REAL NOT NULL,
    active_fraction     REAL NOT NULL,          -- % neurons firing at least once
    logged_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sparsity_exp_epoch ON sparsity_by_layer(experiment_id, epoch);

-- ----------------------------------------------------------------------------
-- View: experiment_summary
-- Convenience view joining latest metrics per experiment.
-- Used by the report generator.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW experiment_summary AS
SELECT
    e.id,
    e.mlflow_run_id,
    e.name,
    e.dataset,
    e.model_variant,
    e.n_patterns,
    e.seed,
    e.status,
    e.started_at,
    e.completed_at,
    em.epoch                    AS final_epoch,
    em.val_accuracy             AS final_accuracy,
    em.val_ece                  AS final_ece,
    em.sparsity                 AS final_sparsity,
    em.epistemic_unc            AS final_epistemic,
    em.aleatoric_unc            AS final_aleatoric,
    en.total_nj                 AS energy_nj,
    en.reduction_vs_gpu         AS energy_reduction_factor
FROM experiments e
LEFT JOIN LATERAL (
    SELECT * FROM epoch_metrics
    WHERE experiment_id = e.id
    ORDER BY epoch DESC LIMIT 1
) em ON TRUE
LEFT JOIN LATERAL (
    SELECT * FROM energy_measurements
    WHERE experiment_id = e.id AND platform = 'loihi2_projected'
    ORDER BY logged_at DESC LIMIT 1
) en ON TRUE;

COMMENT ON VIEW experiment_summary IS
    'Denormalized view for quick experiment comparison and report generation';