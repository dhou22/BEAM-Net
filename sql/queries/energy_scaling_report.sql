-- ============================================================================
-- Query: Energy Scaling Analysis
-- ============================================================================
-- Reconstructs Table 1 from the paper: energy per inference across platforms.
-- Used to validate the 22-30x reduction claim with traceable data.
-- ============================================================================

SELECT
    em.platform,
    em.n_patterns                                   AS N,
    em.d_input                                      AS d,
    em.sparsity_factor                              AS rho,
    em.n_iterations                                 AS L,
    ROUND(em.e_encoding_pj::numeric, 2)            AS encoding_pj,
    ROUND(em.e_coincidence_pj::numeric, 2)         AS coincidence_pj,
    ROUND(em.e_selection_pj::numeric, 2)           AS selection_pj,
    ROUND(em.total_nj::numeric, 3)                 AS total_nj,
    ROUND(em.reduction_vs_gpu::numeric, 2)         AS reduction_factor,
    e.model_variant,
    e.mlflow_run_id
FROM energy_measurements em
JOIN experiments e ON em.experiment_id = e.id
ORDER BY em.n_patterns, em.platform;