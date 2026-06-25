from config import FuzzyTemporalConfig
from data_loader import CatVisitDataLoader
from run_pipeline import (
    run_fuzzy_temporal_pipeline_from_csv,
    run_fuzzy_temporal_pipeline_from_dataframe,
)


# =========================================================
# USAGE EXAMPLE 1: Run from CSV and save all outputs
# =========================================================

def example_run_from_csv_save_outputs():
    input_file = r"C:\Users\shreetam_behera\Downloads\WBID_Dataset\1086044\1086044.csv"
    output_folder = r"C:\Users\shreetam_behera\Downloads\WBID_Dataset\1086044"

    config = FuzzyTemporalConfig(
        root_output_folder=output_folder,
        base_sigma_lb=0.15,
        temporal_weight=0.40,
        state_alpha=0.60,
        decay_minutes=180.0,
        device_col="device_serial",
        use_dynamic_sigma_in_prediction=True,
        use_overlap_penalty_in_prediction=True,
        use_consistency_penalty_in_prediction=True,
    )

    result = run_fuzzy_temporal_pipeline_from_csv(
        input_file=input_file,
        config=config,
        save_outputs=True,
        save_confusion_matrices=True,
    )

    print("Overall accuracy:", result.overall_accuracy)
    print("Confidence algorithm:", result.confidence_algorithm["name"])
    print(result.row_results.head())
    print(result.user_accuracy.head())


# =========================================================
# USAGE EXAMPLE 2: Load CSV separately, then pass DataFrame
# =========================================================

def example_load_dataframe_then_run():
    input_file = r"C:\Users\shreetam_behera\Downloads\WBID_Dataset\1086044\1086044.csv"
    output_folder = r"C:\Users\shreetam_behera\Downloads\WBID_Dataset\1086044"

    config = FuzzyTemporalConfig(root_output_folder=output_folder)

    loader = CatVisitDataLoader(config)
    raw_df = loader.load_csv(input_file)

    result = run_fuzzy_temporal_pipeline_from_dataframe(
        df=raw_df,
        config=config,
        save_outputs=True,
        save_confusion_matrices=True,
    )

    row_results_df = result.row_results
    daily_summary_df = result.daily_summary
    accuracy_df = result.user_accuracy
    inconsistency_df = result.inconsistency_report
    confidence_algorithm = result.confidence_algorithm

    print(row_results_df[[
        "user_id",
        "event_timestamp",
        "recorded_weight",
        "predicted_cat",
        "predicted_name",
        "confidence",
        "confidence_algorithm",
        "correct",
    ]].head())

    print(daily_summary_df.head())
    print(accuracy_df.head())
    print(inconsistency_df.head())
    print(confidence_algorithm)


# =========================================================
# USAGE EXAMPLE 3: In-memory only, no files saved
# =========================================================

def example_in_memory_only():
    input_file = r"C:\Users\shreetam_behera\Downloads\WBID_Dataset\10\10.csv" # 1086044

    config = FuzzyTemporalConfig()
    loader = CatVisitDataLoader(config)
    raw_df = loader.load_csv(input_file)

    result = run_fuzzy_temporal_pipeline_from_dataframe(
        df=raw_df,
        config=config,
        save_outputs=False,
        save_confusion_matrices=False,
    )

    print("Overall accuracy:", result.overall_accuracy)
    print(result.row_results.head())


if __name__ == "__main__":
    # Uncomment one example at a time.
    # example_run_from_csv_save_outputs()
    # example_load_dataframe_then_run()
    example_in_memory_only()
    # pass
