from config import FuzzyTemporalTriangularConfig
from data_loader import CatVisitDataLoader
from run_pipeline import (
    run_triangular_fuzzy_temporal_pipeline_from_csv,
    run_triangular_fuzzy_temporal_pipeline_from_dataframe,
)


def example_run_from_csv_save_outputs():
    input_file = r"C:\Users\shreetam_behera\Downloads\WBID_Dataset\123773\123773.csv"
    output_folder = r"C:\Users\shreetam_behera\Downloads\WBID_Dataset\123773"
    config = FuzzyTemporalTriangularConfig(root_output_folder=output_folder)
    result = run_triangular_fuzzy_temporal_pipeline_from_csv(
        input_file=input_file,
        config=config,
        save_outputs=True,
        save_confusion_matrices=True,
    )
    print("Overall accuracy:", result.overall_accuracy)
    print("Confidence algorithm:", result.confidence_algorithm["name"])
    print(result.row_results.head())
    print(result.inconsistency_report.head())


def example_load_dataframe_then_run():
    input_file = r"C:\Users\shreetam_behera\Downloads\WBID_Dataset\123773\123773.csv"
    output_folder = r"C:\Users\shreetam_behera\Downloads\WBID_Dataset\123773"
    config = FuzzyTemporalTriangularConfig(root_output_folder=output_folder)
    loader = CatVisitDataLoader(config)
    raw_df = loader.load_csv(input_file)
    result = run_triangular_fuzzy_temporal_pipeline_from_dataframe(
        df=raw_df,
        config=config,
        save_outputs=True,
        save_confusion_matrices=True,
    )
    print(result.row_results[[
        "user_id", "event_timestamp", "recorded_weight", "predicted_cat",
        "predicted_name", "confidence", "confidence_algorithm",
        "sigma_selection_method", "correct"
    ]].head())
    print(result.confidence_algorithm)


def example_in_memory_only():
    input_file = r"C:\Users\shreetam_behera\Downloads\WBID_Dataset\123773\123773.csv"
    config = FuzzyTemporalTriangularConfig()
    loader = CatVisitDataLoader(config)
    raw_df = loader.load_csv(input_file)
    result = run_triangular_fuzzy_temporal_pipeline_from_dataframe(
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
