-- ============================================================================
-- Query: Reliability Diagram Data (Calibration Analysis)
-- ============================================================================
-- Bins test predictions by confidence and computes actual accuracy per bin.
-- Output feeds the reliability diagram in the PDF report.
--
-- A perfectly calibrated model has confidence = accuracy in every bin.
-- BEAM-Net's Dirichlet attention (§3.3) should show better alignment
-- than deterministic baselines.
-- ============================================================================

WITH bins AS (
    SELECT
        experiment_id,
        WIDTH_BUCKET(max_probability, 0, 1, 15) AS bin_number,
        COUNT(*) AS n_samples,
        AVG(max_probability) AS mean_confidence,
        AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) AS accuracy
    FROM test_predictions
    WHERE experiment_id = :experiment_id
    GROUP BY experiment_id, bin_number
)
SELECT
    bin_number,
    ROUND((bin_number::numeric - 0.5) / 15, 3) AS bin_center,
    n_samples,
    ROUND(mean_confidence::numeric, 4) AS mean_confidence,
    ROUND(accuracy::numeric, 4)         AS accuracy,
    ROUND(ABS(mean_confidence - accuracy)::numeric, 4) AS calibration_gap
FROM bins
WHERE n_samples > 0
ORDER BY bin_number;