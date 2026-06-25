import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

from config import FuzzyTemporalConfig
from fuzzy_diagnostic import FuzzyDiagnostic


@dataclass
class FuzzyPredictionOutput:
    """Container returned by the predictor."""
    row_results: pd.DataFrame
    daily_summary: pd.DataFrame
    user_accuracy: pd.DataFrame
    inconsistency_report: pd.DataFrame
    overall_accuracy: float
    confidence_algorithm: Dict[str, Any]


class FuzzyPredictor:
    """
    Temporal fuzzy prediction class.

    This class consumes only prepared DataFrames and artifacts created by FuzzyDiagnostic.
    """

    CONFIDENCE_ALGORITHM_NAME = "normalized_fuzzy_temporal_confidence"
    CONFIDENCE_FORMULA = (
        "confidence = max_cat(normalize((1 - temporal_weight) * current_weight_membership "
        "+ temporal_weight * temporal_prior))"
    )

    def __init__(self, config: FuzzyTemporalConfig):
        self.config = config
        self.diagnostic = FuzzyDiagnostic(config)

    @staticmethod
    def normalize_scores(score_dict: Dict[str, float]) -> Dict[str, float]:
        return FuzzyDiagnostic.normalize_scores(score_dict)

    @staticmethod
    def time_decay_factor(delta_minutes, decay_minutes: float) -> float:
        if delta_minutes is None or pd.isna(delta_minutes):
            return 0.0
        return math.exp(-delta_minutes / decay_minutes)

    def confidence_algorithm_metadata(self) -> Dict[str, Any]:
        """Return confidence algorithm details with parameter values."""
        return {
            "name": self.CONFIDENCE_ALGORITHM_NAME,
            "formula": self.CONFIDENCE_FORMULA,
            "description": (
                "Confidence is the normalized combined fuzzy score of the predicted cat. "
                "The combined score uses current fuzzy weight evidence and temporal prior. "
                "Current fuzzy weight evidence may include dynamic sigma, overlap penalty, "
                "and consistency penalty depending on config flags."
            ),
            "parameters": {
                "temporal_weight": self.config.temporal_weight,
                "state_alpha": self.config.state_alpha,
                "decay_minutes": self.config.decay_minutes,
                "use_dynamic_sigma_in_prediction": self.config.use_dynamic_sigma_in_prediction,
                "use_overlap_penalty_in_prediction": self.config.use_overlap_penalty_in_prediction,
                "use_consistency_penalty_in_prediction": self.config.use_consistency_penalty_in_prediction,
                "overlap_penalty_strength": self.config.overlap_penalty_strength,
                "consistency_penalty_strength": self.config.consistency_penalty_strength,
                "base_sigma_lb": self.config.base_sigma_lb,
            },
        }

    def apply_overlap_penalty(
        self,
        scores: Dict[str, float],
        overlap_map: Dict[str, float],
    ) -> Dict[str, float]:
        adjusted = {}
        for cat, score in scores.items():
            overlap = overlap_map.get(cat, 0.0)
            factor = 1.0 - self.config.overlap_penalty_strength * overlap
            factor = max(factor, self.config.min_score_factor)
            adjusted[cat] = score * factor
        return self.normalize_scores(adjusted)

    def apply_consistency_penalty(
        self,
        scores: Dict[str, float],
        consistency_map: Dict[str, float],
    ) -> Dict[str, float]:
        adjusted = {}
        for cat, score in scores.items():
            consistency = consistency_map.get(cat, 1.0)
            if pd.isna(consistency):
                consistency = 1.0
            inconsistency = 1.0 - consistency
            factor = 1.0 - self.config.consistency_penalty_strength * inconsistency
            factor = max(factor, self.config.min_score_factor)
            adjusted[cat] = score * factor
        return self.normalize_scores(adjusted)

    def compute_weight_membership(
        self,
        row: pd.Series,
        catalog: Dict[str, float],
        sigma_map: Optional[Dict[str, float]] = None,
        overlap_map: Optional[Dict[str, float]] = None,
        consistency_map: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute current fuzzy membership for one event."""
        measured_weight_kg = row["scale_weight"]
        raw_scores = {}

        for enroll_id, ref_weight_kg in catalog.items():
            if self.config.use_dynamic_sigma_in_prediction and sigma_map is not None:
                sigma_kg = sigma_map.get(enroll_id, self.config.base_sigma_kg)
            else:
                sigma_kg = self.config.base_sigma_kg

            raw_scores[enroll_id] = self.diagnostic.fuzzy_membership_with_sigma(
                measured_weight_kg=measured_weight_kg,
                reference_weight_kg=ref_weight_kg,
                sigma_kg=sigma_kg,
            )

        scores = self.normalize_scores(raw_scores)

        if self.config.use_overlap_penalty_in_prediction and overlap_map is not None:
            scores = self.apply_overlap_penalty(scores, overlap_map)

        if self.config.use_consistency_penalty_in_prediction and consistency_map is not None:
            scores = self.apply_consistency_penalty(scores, consistency_map)

        return self.normalize_scores(scores)

    def predict_for_day(
        self,
        df_day: pd.DataFrame,
        catalog: Dict[str, float],
        sigma_map: Optional[Dict[str, float]] = None,
        overlap_map: Optional[Dict[str, float]] = None,
        consistency_map: Optional[Dict[str, float]] = None,
        device_col: str = "_device",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Temporal fuzzy prediction for one user, one day, one device."""
        if df_day.empty:
            return pd.DataFrame(), {}

        df_day = df_day.sort_values("event_timestamp").copy()
        cats = list(catalog.keys())

        if len(cats) == 0:
            return pd.DataFrame(), {}

        uniform = 1.0 / len(cats)
        latent_state = {cat: uniform for cat in cats}
        last_timestamp = None

        row_results = []
        cumulative_day_scores = {cat: 0.0 for cat in cats}

        for _, row in df_day.iterrows():
            current_membership = self.compute_weight_membership(
                row=row,
                catalog=catalog,
                sigma_map=sigma_map,
                overlap_map=overlap_map,
                consistency_map=consistency_map,
            )

            if last_timestamp is None:
                gap_minutes = None
                decay = 0.0
            else:
                delta = row["event_timestamp"] - last_timestamp
                gap_minutes = delta.total_seconds() / 60.0
                decay = self.time_decay_factor(gap_minutes, self.config.decay_minutes)

            temporal_prior = {
                cat: decay * latent_state.get(cat, 0.0) + (1.0 - decay) * uniform
                for cat in cats
            }
            temporal_prior = self.normalize_scores(temporal_prior)

            combined = {}
            for cat in cats:
                weight_part = (1.0 - self.config.temporal_weight) * current_membership.get(cat, 0.0)
                temporal_part = self.config.temporal_weight * temporal_prior.get(cat, 0.0)
                combined[cat] = weight_part + temporal_part

            combined = self.normalize_scores(combined)

            updated_state = {}
            for cat in cats:
                updated_state[cat] = (
                    self.config.state_alpha * combined.get(cat, 0.0)
                    + (1.0 - self.config.state_alpha) * latent_state.get(cat, 0.0)
                )
            updated_state = self.normalize_scores(updated_state)

            predicted_cat = max(combined, key=combined.get)
            confidence = combined[predicted_cat]

            predicted_weight_component = (
                (1.0 - self.config.temporal_weight)
                * current_membership.get(predicted_cat, 0.0)
            )
            predicted_temporal_component = (
                self.config.temporal_weight
                * temporal_prior.get(predicted_cat, 0.0)
            )

            for cat in cats:
                cumulative_day_scores[cat] += combined.get(cat, 0.0)

            row_result = {
                "user_id": row["user_id"],
                "device": row[device_col],
                "event_date": row["event_date"],
                "event_timestamp": row["event_timestamp"],
                "recorded_weight": row["scale_weight"],
                "true_label": row["true_label_cat_name"],
                "predicted_cat": predicted_cat,
                "confidence": round(confidence, 6),
                "confidence_algorithm": self.CONFIDENCE_ALGORITHM_NAME,
                "confidence_formula": self.CONFIDENCE_FORMULA,
                "predicted_weight_component": round(predicted_weight_component, 6),
                "predicted_temporal_component": round(predicted_temporal_component, 6),
                "gap_minutes": gap_minutes if gap_minutes is not None else np.nan,
                "temporal_decay": round(decay, 6),
                "used_dynamic_sigma": self.config.use_dynamic_sigma_in_prediction,
                "used_overlap_penalty": self.config.use_overlap_penalty_in_prediction,
                "used_consistency_penalty": self.config.use_consistency_penalty_in_prediction,
            }

            for cat in cats:
                row_result[f"weight_score_{cat}"] = round(current_membership.get(cat, 0.0), 6)
                row_result[f"temporal_score_{cat}"] = round(temporal_prior.get(cat, 0.0), 6)
                row_result[f"combined_score_{cat}"] = round(combined.get(cat, 0.0), 6)

                if sigma_map is not None:
                    row_result[f"dynamic_sigma_kg_{cat}"] = round(sigma_map.get(cat, self.config.base_sigma_kg), 6)
                    row_result[f"dynamic_sigma_lb_{cat}"] = round(
                        sigma_map.get(cat, self.config.base_sigma_kg) * self.config.kg_to_lb, 6
                    )
                if overlap_map is not None:
                    row_result[f"overlap_score_{cat}"] = round(overlap_map.get(cat, 0.0), 6)
                if consistency_map is not None:
                    row_result[f"consistency_score_{cat}"] = round(consistency_map.get(cat, 1.0), 6)

            row_results.append(row_result)
            latent_state = updated_state
            last_timestamp = row["event_timestamp"]

        row_result_df = pd.DataFrame(row_results)

        day_cat = max(cumulative_day_scores, key=cumulative_day_scores.get)
        day_conf = cumulative_day_scores[day_cat] / max(len(df_day), 1)

        day_summary = {
            "user_id": df_day["user_id"].iloc[0],
            "device": df_day[device_col].iloc[0],
            "event_date": df_day["event_date"].iloc[0],
            "dominant_cat": day_cat,
            "dominant_cat_avg_score": round(day_conf, 6),
            "confidence_algorithm": self.CONFIDENCE_ALGORITHM_NAME,
            "events_in_day": len(df_day),
        }

        for cat in cats:
            day_summary[f"day_total_score_{cat}"] = round(cumulative_day_scores.get(cat, 0.0), 6)

        return row_result_df, day_summary

    def run_user_prediction(
        self,
        df_user: pd.DataFrame,
        catalog: Dict[str, float],
        inconsistency_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run temporal prediction for one already-prepared enrolled user dataframe."""
        sigma_map, consistency_map, overlap_map = self.diagnostic.build_prediction_artifacts_from_inconsistency(
            inconsistency_df=inconsistency_df,
            catalog=catalog,
        )

        if self.config.device_col and self.config.device_col in df_user.columns:
            dev_col = self.config.device_col
        else:
            dev_col = "_device"
            df_user = df_user.copy()
            df_user[dev_col] = "single"

        all_day_rows = []
        all_day_summaries = []

        for _, df_day in df_user.groupby(["event_date", dev_col]):
            row_df, day_summary = self.predict_for_day(
                df_day=df_day,
                catalog=catalog,
                sigma_map=sigma_map,
                overlap_map=overlap_map,
                consistency_map=consistency_map,
                device_col=dev_col,
            )

            if not row_df.empty:
                all_day_rows.append(row_df)
            if day_summary:
                all_day_summaries.append(day_summary)

        result_df = pd.concat(all_day_rows, ignore_index=True) if all_day_rows else pd.DataFrame()
        daily_summary_df = pd.DataFrame(all_day_summaries)
        return result_df, daily_summary_df
