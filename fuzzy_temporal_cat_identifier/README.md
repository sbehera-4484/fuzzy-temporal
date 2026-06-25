# Fuzzy Temporal Cat Identifier

This package contains a class-based refactor with only the requested class groups:

1. `FuzzyTemporalConfig` in `config.py`
2. `CatVisitDataLoader` in `data_loader.py`
3. `FuzzyDiagnostic` in `fuzzy_diagnostic.py`
4. `FuzzyPredictor` in `fuzzy_predictor.py`
5. `DebugOutputSaver` in `debug_output_saver.py`

The orchestration layer is intentionally function-based in `run_pipeline.py`, so no extra pipeline/orchestrator class is introduced.

## Main design

CSV path handling is isolated in `CatVisitDataLoader` and `run_fuzzy_temporal_pipeline_from_csv()`.
After loading, the data moves through the pipeline as DataFrames:

```text
CSV path
  -> CatVisitDataLoader.load_csv()
  -> raw DataFrame
  -> CatVisitDataLoader.preprocess_dataframe()
  -> FuzzyDiagnostic
  -> FuzzyPredictor
  -> DebugOutputSaver
```

## Returned outputs

`run_fuzzy_temporal_pipeline_from_dataframe()` and `run_fuzzy_temporal_pipeline_from_csv()` return a `FuzzyPredictionOutput` object with:

- `row_results`
- `daily_summary`
- `user_accuracy`
- `inconsistency_report`
- `overall_accuracy`
- `confidence_algorithm`

## Confidence columns

`row_results` includes:

- `confidence`
- `confidence_algorithm`
- `confidence_formula`
- `predicted_weight_component`
- `predicted_temporal_component`

## Usage

See `usage_examples.py`.
