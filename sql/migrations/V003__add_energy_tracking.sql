-- ============================================================================
-- BEAM-Net: Schema Migration V003 — Enhanced Energy Tracking
-- ============================================================================
-- Purpose:
--   Extends V001's energy_measurements with time-series tracking,
--   hardware platform calibration constants, and scaling benchmark storage.
--
-- Motivation:
--   The energy claim (22-30x reduction vs GPU, Table 1 in paper) requires
--   traceable provenance: which hardware constants were used, which N values
--   were tested, which sparsity factor rho was measured vs assumed.
--
-- Tables added:
--   - hardware_platforms : Reference table of E_spike constants per platform
--   - energy_timeseries  : Per-batch energy during training (not just final)
--   - energy_scaling     : Dedicated storage for Eq. 25 scaling sweeps
-- ============================================================================

\c beam_metrics;

-- ----------------------------------------------------------------------------
-- Table: hardware_platforms
-- Reference data for energy calculations. Populated once, queried on every
-- energy computation. Source citations embedded for CIFRE audit trail.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hardware_platforms (
    platform_id         VARCHAR(32) PRIMARY KEY,
    platform_name       VARCHAR(64) NOT NULL,
    e_spike_pj          REAL,                   -- picojoules per spike op
    e_mac_pj            REAL,                   -- picojoules per multiply-accumulate
    idle_power_w        REAL,
    peak_power_w        REAL,
    source_citation     TEXT NOT NULL,          -- Peer-reviewed source
    measured_or_projected VARCHAR(16) NOT NULL DEFAULT 'projected',
    added_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_measurement_type CHECK (measured_or_projected IN ('measured', 'projected'))
);

COMMENT ON TABLE hardware_platforms IS
    'Reference constants for energy calculations. Every value traceable to peer-reviewed source.';

-- Seed with literature values
INSERT INTO hardware_platforms
    (platform_id, platform_name, e_spike_pj, e_mac_pj, idle_power_w, peak_power_w,
     source_citation, measured_or_projected)
VALUES
    ('loihi2',
     'Intel Loihi 2 Neuromorphic Processor',
     23.6, NULL, 0.02, 0.1,
     'Davies M et al. (2021). Advancing neuromorphic computing with Loihi. Proc. IEEE 109(5):911-934.',
     'measured'),
    ('gpu_a100',
     'NVIDIA A100 Tensor Core GPU',
     NULL, 0.3, 50.0, 300.0,
     'NVIDIA A100 Datasheet (2020). Estimated 0.3 pJ/FP16 MAC.',
     'measured'),
    ('spikformer',
     'SpikFormer on neuromorphic substrate (reference)',
     25.0, NULL, 0.5, 1.5,
     'Zhou Z et al. (2023). Spikformer: When spiking neural network meets transformer. ICLR.',
     'projected'),
    ('biological',
     'Cortical attention circuit (reference)',
     NULL, NULL, NULL, 0.02,
     'Lennie P (2003). The cost of cortical computation. Curr. Biol. 13(6):493-497.',
     'measured')
ON CONFLICT (platform_id) DO NOTHING;

-- ----------------------------------------------------------------------------
-- Table: energy_timeseries
-- Per-batch energy accounting during training runs.
-- Enables analysis of energy scaling with training progress (e.g., does
-- sparsity growth during training reduce per-batch energy over time?).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS energy_timeseries (
    id                  BIGSERIAL PRIMARY KEY,
    experiment_id       INT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    platform_id         VARCHAR(32) NOT NULL REFERENCES hardware_platforms(platform_id),
    epoch               INT NOT NULL,
    batch_idx           INT NOT NULL,
    measured_sparsity   REAL,                   -- Actual rho from this batch
    n_spikes_emitted    BIGINT,                 -- Actual spike count
    energy_nj           REAL NOT NULL,          -- Per-sample energy estimate
    batch_size          INT NOT NULL,
    logged_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_energy_ts_exp ON energy_timeseries(experiment_id, epoch, batch_idx);
CREATE INDEX idx_energy_ts_platform ON energy_timeseries(platform_id);

COMMENT ON TABLE energy_timeseries IS
    'Per-batch energy measurements during training. Supports dynamic sparsity analysis.';

-- ----------------------------------------------------------------------------
-- Table: energy_scaling
-- Dedicated storage for scaling sweep experiments (Eq. 25).
-- Each row = one (N, d, rho, L, platform) tuple benchmark.
-- Used to reproduce Figure 2 / Table 1 from the paper.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS energy_scaling (
    id                  BIGSERIAL PRIMARY KEY,
    experiment_id       INT REFERENCES experiments(id) ON DELETE CASCADE,
    platform_id         VARCHAR(32) NOT NULL REFERENCES hardware_platforms(platform_id),
    -- Eq. 25 parameters
    n_patterns          INT NOT NULL,           -- N
    d_input             INT NOT NULL,           -- d
    sparsity_factor     REAL NOT NULL,          -- rho
    n_iterations        INT NOT NULL,           -- L
    -- Eq. 25 component breakdown (picojoules)
    e_encoding_pj       REAL NOT NULL,          -- d · E_spike
    e_coincidence_pj    REAL NOT NULL,          -- (d + rho·d·N) · E_spike
    e_selection_pj      REAL NOT NULL,          -- (1.1·N² + L·N) · E_spike
    -- Derived
    total_pj            REAL GENERATED ALWAYS AS
                        (e_encoding_pj + e_coincidence_pj + e_selection_pj) STORED,
    total_nj            REAL NOT NULL,
    -- Context
    benchmark_type      VARCHAR(32) NOT NULL DEFAULT 'theoretical',
    notes               TEXT,
    logged_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_benchmark CHECK (benchmark_type IN ('theoretical', 'simulated', 'measured'))
);

CREATE INDEX idx_scaling_n ON energy_scaling(n_patterns);
CREATE INDEX idx_scaling_platform ON energy_scaling(platform_id, n_patterns);

COMMENT ON TABLE energy_scaling IS
    'Scaling benchmarks per Eq. 25. Stores N × d × rho × L sweeps for paper Table 1 reproduction.';

-- ----------------------------------------------------------------------------
-- View: energy_comparison
-- Joins energy_scaling with platforms for direct reduction-factor computation.
-- Reproduces Table 1 format from the paper.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW energy_comparison AS
WITH gpu_baseline AS (
    SELECT n_patterns, d_input, total_nj AS gpu_nj
    FROM energy_scaling
    WHERE platform_id = 'gpu_a100'
)
SELECT
    es.n_patterns           AS N,
    es.d_input              AS d,
    hp.platform_name,
    es.sparsity_factor      AS rho,
    es.n_iterations         AS L,
    ROUND(es.total_nj::numeric, 3)          AS energy_nj,
    ROUND((gb.gpu_nj / es.total_nj)::numeric, 2) AS reduction_vs_gpu,
    hp.source_citation
FROM energy_scaling es
JOIN hardware_platforms hp ON es.platform_id = hp.platform_id
LEFT JOIN gpu_baseline gb ON es.n_patterns = gb.n_patterns AND es.d_input = gb.d_input
ORDER BY es.n_patterns, hp.platform_name;

COMMENT ON VIEW energy_comparison IS
    'Paper Table 1 reproduction. Shows energy per platform with reduction factor vs GPU baseline.';

-- ----------------------------------------------------------------------------
-- Trigger: auto-populate experiments.completed_at when status changes
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_completed_at()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status IN ('complete', 'failed') AND OLD.status = 'running' THEN
        NEW.completed_at = CURRENT_TIMESTAMP;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_set_completed_at ON experiments;
CREATE TRIGGER trg_set_completed_at
    BEFORE UPDATE ON experiments
    FOR EACH ROW
    EXECUTE FUNCTION set_completed_at();