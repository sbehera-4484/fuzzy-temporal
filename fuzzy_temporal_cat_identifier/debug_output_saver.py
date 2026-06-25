import os
from typing import Dict, Any

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import FuzzyTemporalConfig
from fuzzy_predictor import FuzzyPredictionOutput


class DebugOutputSaver:
    """
    Debugging/output helper class.

    Responsibilities:
    - create output folders
    - save prediction/debug CSV files
    - save confusion matrix PNG/CSV
    - save confidence algorithm metadata
    """

    def __init__(self, config: FuzzyTemporalConfig):
        self.config = config

    def ensure_folder(self, path: str | None = None) -> None:
        output_path = path or self.config.root_output_folder
        if output_path:
            os.makedirs(output_path, exist_ok=True)

    @staticmethod
    def save_placeholder_png(output_path: str, title: str, message: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"{title}\n\n{message}",
            ha="center",
            va="center",
            fontsize=12,
            wrap=True,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def save_placeholder_cm_csv(output_path: str, message: str) -> None:
        pd.DataFrame({"message": [message]}).to_csv(output_path, index=False)

    def save_confusion_matrix_outputs(
        self,
        y_true,
        y_pred,
        png_path: str,
        csv_path: str,
        title: str,
    ) -> str:
        """Save confusion matrix as PNG and CSV."""
        try:
            df_cm = pd.DataFrame({"true": y_true, "pred": y_pred}).dropna()

            if df_cm.empty:
                msg = "No valid data available for confusion matrix."
                self.save_placeholder_png(png_path, title, msg)
                self.save_placeholder_cm_csv(csv_path, msg)
                return "placeholder_empty"

            labels = sorted(list(set(df_cm["true"]) | set(df_cm["pred"])))

            if len(labels) < 2:
                msg = "Only one class present. Confusion matrix not meaningful."
                self.save_placeholder_png(png_path, title, msg)
                self.save_placeholder_cm_csv(csv_path, msg)
                return "placeholder_single_class"

            cm = confusion_matrix(df_cm["true"], df_cm["pred"], labels=labels)
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            cm_df.index.name = "true_label"
            cm_df.to_csv(csv_path)

            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(png_path, dpi=200)
            plt.close()

            return "saved"

        except Exception as e:
            msg = f"Error while generating confusion matrix: {str(e)}"
            self.save_placeholder_png(png_path, title, msg)
            self.save_placeholder_cm_csv(csv_path, msg)
            return f"error: {e}"

    def save_confidence_algorithm(
        self,
        confidence_algorithm: Dict[str, Any],
        output_path: str,
    ) -> None:
        """Save confidence algorithm metadata as CSV."""
        records = []

        for key, value in confidence_algorithm.items():
            if key == "parameters":
                continue
            records.append({"key": key, "value": value})

        for key, value in confidence_algorithm.get("parameters", {}).items():
            records.append({"key": f"parameter.{key}", "value": value})

        pd.DataFrame(records).to_csv(output_path, index=False)

    def save_all_outputs(self, output: FuzzyPredictionOutput) -> Dict[str, str]:
        """Save final DataFrames and metadata to configured output folder."""
        if not self.config.root_output_folder:
            raise ValueError("root_output_folder must be configured before saving.")

        self.ensure_folder()

        output_paths = {
            "row_results": os.path.join(self.config.root_output_folder, "fsc_temporal_fuzzy_row_results.csv"),
            "daily_summary": os.path.join(self.config.root_output_folder, "fsc_daily_temporal_cat_identity.csv"),
            "user_accuracy": os.path.join(self.config.root_output_folder, "fsc_user_accuracy.csv"),
            "inconsistency_report": os.path.join(self.config.root_output_folder, "fsc_cat_inconsistency_report.csv"),
            "confidence_algorithm": os.path.join(self.config.root_output_folder, "fsc_confidence_algorithm.csv"),
        }

        output.row_results.to_csv(output_paths["row_results"], index=False)
        output.daily_summary.to_csv(output_paths["daily_summary"], index=False)
        output.user_accuracy.to_csv(output_paths["user_accuracy"], index=False)
        output.inconsistency_report.to_csv(output_paths["inconsistency_report"], index=False)
        self.save_confidence_algorithm(output.confidence_algorithm, output_paths["confidence_algorithm"])

        return output_paths
