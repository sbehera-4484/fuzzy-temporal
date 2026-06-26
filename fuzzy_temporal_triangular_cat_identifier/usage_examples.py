import os
import pandas as pd
from config import FuzzyTemporalTriangularConfig
from data_loader import CatVisitDataLoader
from run_pipeline import (
    run_triangular_fuzzy_temporal_pipeline_from_csv,
    run_triangular_fuzzy_temporal_pipeline_from_dataframe,
)


def example_run_with_abstention_enabled():
    # Replace these paths with your local dataset paths
    input_file = r"C:\Users\shreetam_behera\Downloads\WBID_Dataset\123773\123773.csv"
    output_folder = r"C:\Users\shreetam_behera\Downloads\WBID_Dataset\123773"
    
    # 1. Initialize configuration with custom abstention thresholds
    config = FuzzyTemporalTriangularConfig(
        root_output_folder=output_folder,
        use_abstention=True,             # Turn on the newly added abstention rules
        abstain_label="unknown",         # Text label used when a prediction is skipped
        confidence_threshold=0.50,       # Minimum required combined confidence score
        overlap_abstention_threshold=0.70, # Max allowed fuzzy overlap between cat profiles
        score_gap_threshold=0.10         # Minimum distance between top 1 and top 2 choices
    )
    
    print("--- Running Pipeline from CSV ---")
    # 2. Run the pipeline execution engine
    result = run_triangular_fuzzy_temporal_pipeline_from_csv(
        input_file=input_file,
        config=config,
        save_outputs=True,
        save_confusion_matrices=True,
    )
    
    # 3. Inspect global performance and coverage metrics
    print("\n=== Global Metrics ===")
    print(f"Overall Accuracy (All Events):   {result.overall_accuracy:.4f}")
    print(f"Selective Accuracy (Decided):    {result.selective_accuracy:.4f}")
    print(f"System Coverage:                 {result.coverage * 100:.2f}%")
    print(f"Abstention Rate:                 {result.abstention_rate * 100:.2f}%")
    
    # 4. View raw timeline log metrics
    print("\n=== Sample Predictions Timeline ===")
    cols_to_show = [
        "user_id", "event_timestamp", "recorded_weight", "true_label",
        "raw_predicted_cat", "predicted_cat", "predicted_name",
        "abstained", "abstention_reason", "confidence",
        "top1_top2_gap", "predicted_overlap_for_abstention", "correct"
    ]
    print(result.row_results[cols_to_show].head(10))
    
    # 5. Look at user-level metrics table
    print("\n=== User Summary Performance ===")
    print(result.user_accuracy.head())

    if result.abstention_summary is not None:
        print("\n=== Abstention Summary ===")
        print(result.abstention_summary.head())

    return result


def example_in_memory_dataframe_run():
    # Example showing how to pass a pre-loaded pandas DataFrame manually
    input_file = r"C:\Users\shreetam_behera\Downloads\WBID_Dataset\123773\123773.csv"

    config = FuzzyTemporalTriangularConfig(
        use_abstention=True,
        confidence_threshold=0.50,
        overlap_abstention_threshold=0.70,
        score_gap_threshold=0.10,
    )

    loader = CatVisitDataLoader(config)
    
    # Pre-load data in-memory
    raw_df = loader.load_csv(input_file)
    
    print("\n--- Running Pipeline from In-Memory DataFrame ---")
    result = run_triangular_fuzzy_temporal_pipeline_from_dataframe(
        df=raw_df,
        config=config,
        save_outputs=False,
        save_confusion_matrices=False,
    )

    print(f"Selective accuracy on decided events: {result.selective_accuracy:.4f}")
    print(f"Coverage: {result.coverage * 100:.2f}%")
    print(f"Abstention rate: {result.abstention_rate * 100:.2f}%")

    return result


def test_abstention_logic_with_synthetic_data():
    """
    Test function for validating abstention logic without external files.

    This synthetic test creates a two-cat household with very close weights.
    Because the measurements are between the two profiles, the model should
    produce low gap / high overlap situations and abstain when thresholds are strict.
    """
    synthetic_df = pd.DataFrame({
        "user_id": [999001] * 10,
        "visit_date": ["2026-01-01"] * 10,
        "visit_time": [
            "08:00:00", "08:10:00", "08:20:00", "08:30:00", "08:40:00",
            "09:00:00", "09:10:00", "09:20:00", "09:30:00", "09:40:00",
        ],
        "device_serial": ["device_test"] * 10,
        "profile_weight": [4.00, 4.08] * 5,
        "scale_weight": [4.03, 4.04, 4.05, 4.02, 4.04, 4.05, 4.03, 4.04, 4.02, 4.05],
        "true_label_cat_name": ["cat_A", "cat_B"] * 5,
    })

    config = FuzzyTemporalTriangularConfig(
        use_abstention=True,
        abstain_label="unknown",
        confidence_threshold=0.55,
        overlap_abstention_threshold=0.60,
        score_gap_threshold=0.15,
        temporal_weight=0.40,
        state_alpha=0.60,
        decay_minutes=180.0,
    )

    result = run_triangular_fuzzy_temporal_pipeline_from_dataframe(
        df=synthetic_df,
        config=config,
        save_outputs=False,
        save_confusion_matrices=False,
    )

    print("\n[test] Abstention logic synthetic test")
    print("Overall accuracy:", result.overall_accuracy)
    print("Selective accuracy:", result.selective_accuracy)
    print("Coverage:", result.coverage)
    print("Abstention rate:", result.abstention_rate)

    columns_to_show = [
        "event_timestamp",
        "recorded_weight",
        "raw_predicted_cat",
        "predicted_cat",
        "predicted_name",
        "confidence",
        "top1_top2_gap",
        "predicted_overlap_for_abstention",
        "abstained",
        "abstention_reason",
        "correct_raw",
        "correct",
    ]
    print(result.row_results[columns_to_show])

    print("\n[test] Abstention summary")
    print(result.abstention_summary)

    assert "abstained" in result.row_results.columns
    assert "abstention_reason" in result.row_results.columns
    assert "raw_predicted_cat" in result.row_results.columns
    assert "predicted_cat" in result.row_results.columns

    return result


if __name__ == "__main__":
    # Uncomment one example at a time.
    # example_run_with_abstention_enabled()
    example_in_memory_dataframe_run()
    test_abstention_logic_with_synthetic_data()
