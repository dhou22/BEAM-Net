-- ============================================================================
-- Query: Model Variant Comparison
-- ============================================================================
-- Compares BEAM-Net vs baselines on a specified dataset.
-- Used by report_generator.py to populate comparison tables.
--
-- Usage:
--   psql -d beam_metrics -v dataset="'nmnist'" -f queries/experiment_comparison.sql
-- ============================================================================

SELECT
    model_variant,
    COUNT(*)                              AS n_runs,
    ROUND(AVG(final_accuracy)::numeric, 4) AS mean_accuracy,
    ROUND(STDDEV(final_accuracy)::numeric, 4) AS std_accuracy,
    ROUND(AVG(final_ece)::numeric, 4)      AS mean_ece,
    ROUND(AVG(final_sparsity)::numeric, 4) AS mean_sparsity,
    ROUND(AVG(energy_nj)::numeric, 2)     AS mean_energy_nj,
    ROUND(AVG(energy_reduction_factor)::numeric, 2) AS mean_reduction_vs_gpu
FROM experiment_summary
WHERE dataset = :dataset
  AND status = 'complete'
GROUP BY model_variant
ORDER BY mean_accuracy DESC;