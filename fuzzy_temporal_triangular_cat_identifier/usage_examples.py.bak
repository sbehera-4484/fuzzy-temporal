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
        "raw_predicted_cat", "predicted_cat", "abstained", "abstention_reason", "confidence"
    ]
    print(result.row_results[cols_to_show].head(10))
    
    # 5. Look at user-level metrics table
    print("\n=== User Summary Performance ===")
    print(result.user_accuracy.head())


def example_in_memory_dataframe_run():
    # Example showing how to pass a pre-loaded pandas DataFrame manually
    input_file = r"C:\Users\shreetam_behera\Downloads\WBID_Dataset\123773\123773.csv"
    
    config = FuzzyTemporalTriangularConfig(use_abstention=True)
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
    print(f"Selective Accuracy on Decided Events: {result.selective_accuracy:.4f}")


if __name__ == "__main__":
    # Ensure folders exist or change paths above before running
    try:
        example_run_with_abstention_enabled()
        # example_in_memory_dataframe_run()
    except FileNotFoundError:
        print("[Warning] Please update the hardcoded file paths in usage_examples.py to match your environment.")