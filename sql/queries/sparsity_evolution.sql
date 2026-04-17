-- ============================================================================
-- Query: Sparsity Evolution
-- ============================================================================
-- Extracts sparsity over training iterations for a given experiment.
-- Replicates the W-TCRL Figure 6 analysis (Fois & Girau, 2023) for comparison.
--
-- Interpretation:
--   - Decreasing sparsity = homeostatic mechanism at work (cf. W-TCRL Eq. 5)
--   - Plateau around 0.01-0.05 = biologically realistic range (Lennie, 2003)
-- ============================================================================

SELECT
    e.id                AS experiment_id,
    e.model_variant,
    e.n_patterns,
    s.epoch,
    s.layer_name,
    ROUND(s.active_fraction::numeric, 4)  AS active_fraction,
    ROUND(s.spike_count_mean::numeric, 3) AS mean_spikes_per_neuron
FROM sparsity_by_layer s
JOIN experiments e ON s.experiment_id = e.id
WHERE e.mlflow_run_id = :run_id
ORDER BY s.epoch, s.layer_name;