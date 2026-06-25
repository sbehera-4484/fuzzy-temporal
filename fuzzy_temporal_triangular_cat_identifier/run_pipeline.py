import os
import pandas as pd
from config import FuzzyTemporalTriangularConfig
from data_loader import CatVisitDataLoader
from fuzzy_diagnostic import FuzzyDiagnostic
from fuzzy_predictor import FuzzyPredictor, FuzzyPredictionOutput
from debug_output_saver import DebugOutputSaver


def run_triangular_fuzzy_temporal_pipeline_from_dataframe(
    df: pd.DataFrame,
    config: FuzzyTemporalTriangularConfig,
    save_outputs: bool = False,
    save_confusion_matrices: bool = True,
) -> FuzzyPredictionOutput:
    """Function-based orchestrator. No extra pipeline class is used."""
    loader = CatVisitDataLoader(config)
    diagnostic = FuzzyDiagnostic(config)
    predictor = FuzzyPredictor(config)
    saver = DebugOutputSaver(config)
    prepared_df = loader.preprocess_dataframe(df)
    unique_users = prepared_df["user_id"].dropna().unique()
    if save_outputs or save_confusion_matrices:
        if not config.root_output_folder:
            raise ValueError("root_output_folder is required when saving outputs/confusion matrices.")
        saver.ensure_folder()
    all_results, all_daily_summaries, all_inconsistencies, accuracy_records, cm_status_records = [], [], [], [], []
    for user_id in unique_users:
        df_user = loader.get_user_dataframe(prepared_df, user_id)
        if df_user.empty:
            continue
        df_user = loader.add_enrollment_id(df_user)
        catalog = loader.build_enrollment_catalog(df_user)
        inconsistency_df = diagnostic.analyze_cat_inconsistency_fuzzy(df_user, catalog)
        row_df, daily_summary_df = predictor.run_user_prediction(df_user, catalog, inconsistency_df)
        accuracy = None
        if not row_df.empty:
            name_map = loader.map_enrollment_to_names(df_user)
            row_df["predicted_name"] = row_df["predicted_cat"].map(name_map)
            row_df["correct"] = row_df["predicted_name"] == row_df["true_label"]
            accuracy = row_df["correct"].mean()
            if not daily_summary_df.empty:
                daily_summary_df["dominant_name"] = daily_summary_df["dominant_cat"].map(name_map)
            all_results.append(row_df)
            if not daily_summary_df.empty:
                all_daily_summaries.append(daily_summary_df)
            accuracy_records.append({"user_id": user_id, "accuracy": round(accuracy, 6) if accuracy is not None else None})
        if inconsistency_df is not None and not inconsistency_df.empty:
            inconsistency_df = inconsistency_df.copy()
            inconsistency_df["user_id"] = user_id
            all_inconsistencies.append(inconsistency_df)
        if save_confusion_matrices:
            user_cm_png = os.path.join(config.root_output_folder, f"user_{user_id}_confusion_matrix.png")
            user_cm_csv = os.path.join(config.root_output_folder, f"user_{user_id}_confusion_matrix_values.csv")
            if not row_df.empty:
                status = saver.save_confusion_matrix_outputs(row_df["true_label"], row_df["predicted_name"], user_cm_png, user_cm_csv, f"Temporal Confusion Matrix - User {user_id}")
            else:
                msg = "No prediction data available for this user."
                saver.save_placeholder_png(user_cm_png, f"Temporal Confusion Matrix - User {user_id}", msg)
                saver.save_placeholder_cm_csv(user_cm_csv, msg)
                status = "placeholder_no_prediction_data"
            cm_status_records.append({"user_id": user_id, "png_path": user_cm_png, "csv_path": user_cm_csv, "status": status})
    final_row_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    final_daily_summary = pd.concat(all_daily_summaries, ignore_index=True) if all_daily_summaries else pd.DataFrame()
    user_accuracy_df = pd.DataFrame(accuracy_records) if accuracy_records else pd.DataFrame(columns=["user_id", "accuracy"])
    inconsistency_report_df = pd.concat(all_inconsistencies, ignore_index=True) if all_inconsistencies else pd.DataFrame()
    overall_accuracy = float(final_row_results["correct"].mean()) if not final_row_results.empty else 0.0
    output = FuzzyPredictionOutput(
        row_results=final_row_results,
        daily_summary=final_daily_summary,
        user_accuracy=user_accuracy_df,
        inconsistency_report=inconsistency_report_df,
        overall_accuracy=overall_accuracy,
        confidence_algorithm=predictor.confidence_algorithm_metadata(),
    )
    if save_outputs:
        saver.save_all_outputs(output)
    if save_confusion_matrices and config.root_output_folder:
        pd.DataFrame(cm_status_records).to_csv(os.path.join(config.root_output_folder, "confusion_matrix_status.csv"), index=False)
    return output


def run_triangular_fuzzy_temporal_pipeline_from_csv(
    input_file: str,
    config: FuzzyTemporalTriangularConfig,
    save_outputs: bool = False,
    save_confusion_matrices: bool = True,
) -> FuzzyPredictionOutput:
    loader = CatVisitDataLoader(config)
    raw_df = loader.load_csv(input_file)
    return run_triangular_fuzzy_temporal_pipeline_from_dataframe(raw_df, config, save_outputs, save_confusion_matrices)
