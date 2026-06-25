# Fuzzy Temporal Cat Identifier - Triangular Sigma Version

This refactor keeps only the requested class groups:

1. `FuzzyTemporalTriangularConfig` in `config.py`
2. `CatVisitDataLoader` in `data_loader.py`
3. `FuzzyDiagnostic` in `fuzzy_diagnostic.py`
4. `FuzzyPredictor` in `fuzzy_predictor.py`
5. `DebugOutputSaver` in `debug_output_saver.py`

The orchestration layer is function-based in `run_pipeline.py`; no extra pipeline/orchestrator class is introduced.

## Data flow

```text
CSV path
  -> CatVisitDataLoader.load_csv()
  -> raw DataFrame
  -> CatVisitDataLoader.preprocess_dataframe()
  -> FuzzyDiagnostic with triangular sigma-band selection
  -> FuzzyPredictor
  -> DebugOutputSaver
```

## Returned output

`run_triangular_fuzzy_temporal_pipeline_from_dataframe()` and `run_triangular_fuzzy_temporal_pipeline_from_csv()` return `FuzzyPredictionOutput` with:

- `row_results`
- `daily_summary`
- `user_accuracy`
- `inconsistency_report`
- `overall_accuracy`
- `confidence_algorithm`

## Triangular-specific outputs

`inconsistency_report` includes:

- `dominant_sigma_band`
- `mu_sigma_0_5_to_1_0_lb`
- `mu_sigma_1_0_to_2_0_lb`
- `mu_sigma_2_0_to_5_0_lb`
- `mu_sigma_ge_5_0_lb`

`row_results` includes:

- `confidence`
- `confidence_algorithm`
- `confidence_formula`
- `predicted_weight_component`
- `predicted_temporal_component`
- `sigma_selection_method`

See `usage_examples.py`.
