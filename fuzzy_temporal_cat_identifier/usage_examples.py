import os
import pandas as pd

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
        use_abstention=True,
        confidence_threshold=0.50,
        overlap_abstention_threshold=0.70,
        score_gap_threshold=0.10,
        abstain_label="unknown",
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
    print("Selective accuracy on non-abstained events:", result.selective_accuracy)
    print("Coverage:", result.coverage)
    print("Abstention rate:", result.abstention_rate)
    print(result.row_results.head())
    print(result.abstention_summary.head())
    print(result.user_accuracy.head())


# =========================================================
# USAGE EXAMPLE 1B: Run from CSV for ALL users in subfolders
# =========================================================

def example_run_from_csv_save_outputs_all_users():
    """
    Processes all user_ids stored as subfolders.
    Each subfolder contains a CSV file with the same name as the folder.
    """
    base_folder = r"C:\Users\shreetam_behera\Downloads\WBID_Scenarios\WBID_MultipleCats_Overlapping_Dataset"
    
    # Get all subfolders
    try:
        subfolders = [f for f in os.listdir(base_folder) 
                      if os.path.isdir(os.path.join(base_folder, f))]
    except FileNotFoundError:
        print(f"Base folder not found: {base_folder}")
        return
    
    print(f"Found {len(subfolders)} user folders to process\n")
    
    results_summary = []
    
    for user_folder in sorted(subfolders):
        user_id = user_folder
        input_file = os.path.join(base_folder, user_folder, f"{user_folder}.csv")
        output_folder = os.path.join(base_folder, user_folder)
        
        # Check if CSV file exists
        if not os.path.exists(input_file):
            print(f"⚠️  Skipping {user_id}: {input_file} not found")
            continue
        
        print(f"Processing user {user_id}...")
        
        try:
            config = FuzzyTemporalConfig(
                root_output_folder=output_folder,
                base_sigma_lb=0.15,
                temporal_weight=0.40,
                state_alpha=0.60,
                decay_minutes=180.0,
                device_col="device_serial",
                use_abstention=False,
                confidence_threshold=0.50,
                overlap_abstention_threshold=0.70,
                score_gap_threshold=0.10,
                abstain_label="unknown",
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

            # Store summary results
            results_summary.append({
                "user_id": user_id,
                "overall_accuracy": result.overall_accuracy,
                "selective_accuracy": result.selective_accuracy,
                "coverage": result.coverage,
                "abstention_rate": result.abstention_rate,
                "total_events": len(result.row_results),
                "abstained_events": result.row_results["abstained"].sum(),
            })
            
            print(f"  ✓ Overall accuracy: {result.overall_accuracy:.4f}")
            print(f"  ✓ Selective accuracy: {result.selective_accuracy:.4f}")
            print(f"  ✓ Coverage: {result.coverage:.4f}")
            print(f"  ✓ Abstention rate: {result.abstention_rate:.4f}\n")
            
        except Exception as e:
            print(f"  ✗ Error processing user {user_id}: {str(e)}\n")
            continue
    
    # Print summary table
    if results_summary:
        print("\n" + "="*80)
        print("SUMMARY OF ALL USERS")
        print("="*80)
        summary_df = pd.DataFrame(results_summary)
        print(summary_df.to_string(index=False))
        print(f"\nMean overall accuracy: {summary_df['overall_accuracy'].mean():.4f}")
        print(f"Mean selective accuracy: {summary_df['selective_accuracy'].mean():.4f}")
        print(f"Mean coverage: {summary_df['coverage'].mean():.4f}")


# =========================================================
# USAGE EXAMPLE 2: Load CSV separately, then pass DataFrame
# =========================================================

def example_load_dataframe_then_run():
    input_file = r"C:\Users\shreetam_behera\Downloads\WBID_Dataset\1086044\1086044.csv"
    output_folder = r"C:\Users\shreetam_behera\Downloads\WBID_Dataset\1086044"

    config = FuzzyTemporalConfig(
        root_output_folder=output_folder,
        use_abstention=True,
        confidence_threshold=0.55,
        overlap_abstention_threshold=0.70,
        score_gap_threshold=0.08,
    )
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
		"raw_predicted_cat",			
        "predicted_cat",
        "predicted_name",
        "confidence",
        "confidence_algorithm",
        "top1_top2_gap",
        "predicted_overlap_for_abstention",
        "abstained",
        "abstention_reason",
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

    #config = FuzzyTemporalConfig()
    config = FuzzyTemporalConfig(
        use_abstention=True,
        confidence_threshold=0.50,
        overlap_abstention_threshold=0.70,
        score_gap_threshold=0.10,
    )    
    loader = CatVisitDataLoader(config)
    raw_df = loader.load_csv(input_file)

    result = run_fuzzy_temporal_pipeline_from_dataframe(
        df=raw_df,
        config=config,
        save_outputs=False,
        save_confusion_matrices=False,
    )

    print("Overall accuracy with abstention:", result.overall_accuracy)
    print("Selective accuracy:", result.selective_accuracy)
    print("Coverage:", result.coverage)
    print("Abstention rate:", result.abstention_rate)
    print(result.row_results.head())

# =========================================================
# TEST FUNCTION: Synthetic abstention validation
# =========================================================

def test_abstention_logic_with_synthetic_data():
    """
    Small synthetic test for abstention logic.

    What this test does:
    1. Builds a two-cat household with close profile weights.
    2. Uses ambiguous scale weights between both cats.
    3. Runs the full pipeline in memory.
    4. Prints abstention-related columns.

    Expected behavior:
    - At least some rows should abstain if thresholds are strict enough.
    - You can tune confidence_threshold, overlap_abstention_threshold, and score_gap_threshold.
    """

    synthetic_df = pd.DataFrame({
        "user_id": [999001] * 8,
        "visit_date": ["2026-01-01"] * 8,
        "visit_time": [
            "08:00:00",
            "08:10:00",
            "08:20:00",
            "08:30:00",
            "09:00:00",
            "09:10:00",
            "09:20:00",
            "09:30:00",
        ],
        "device_serial": ["device_test"] * 8,
        "profile_weight": [4.00, 4.08, 4.00, 4.08, 4.00, 4.08, 4.00, 4.08],
        "scale_weight": [4.03, 4.04, 4.05, 4.02, 4.04, 4.05, 4.03, 4.04],
        "true_label_cat_name": ["cat_A", "cat_B", "cat_A", "cat_B", "cat_A", "cat_B", "cat_A", "cat_B"],
    })

    config = FuzzyTemporalConfig(
        use_abstention=True,
        confidence_threshold=0.55,
        overlap_abstention_threshold=0.60,
        score_gap_threshold=0.15,
        abstain_label="unknown",
        temporal_weight=0.40,
        state_alpha=0.60,
        decay_minutes=180.0,
    )

    result = run_fuzzy_temporal_pipeline_from_dataframe(
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
    # example_run_from_csv_save_outputs()
    example_run_from_csv_save_outputs_all_users()  # Process all users in subfolders
    # example_load_dataframe_then_run()
    # example_in_memory_only()
    # test_abstention_logic_with_synthetic_data()
