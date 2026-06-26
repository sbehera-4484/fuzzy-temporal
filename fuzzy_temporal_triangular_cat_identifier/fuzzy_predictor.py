import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from config import FuzzyTemporalTriangularConfig
from fuzzy_diagnostic import FuzzyDiagnostic


@dataclass
class FuzzyPredictionOutput:
    row_results: pd.DataFrame
    daily_summary: pd.DataFrame
    user_accuracy: pd.DataFrame
    inconsistency_report: pd.DataFrame
    overall_accuracy: float
    selective_accuracy: float  # Added metric
    coverage: float            # Added metric
    abstention_rate: float     # Added metric
    confidence_algorithm: Dict[str, Any]
    abstention_algorithm: Dict[str, Any]
    abstention_summary: pd.DataFrame | None = None


class FuzzyPredictor:
    """Temporal fuzzy predictor using triangular-selected dynamic sigma artifacts with abstention."""
    CONFIDENCE_ALGORITHM_NAME = "normalized_fuzzy_temporal_confidence_with_triangular_sigma"
    CONFIDENCE_FORMULA = (
        "confidence = max_cat(normalize((1 - temporal_weight) * current_weight_membership "
        "+ temporal_weight * temporal_prior))"
    )
    ABSTENTION_ALGORITHM_NAME = "combined_confidence_overlap_gap_abstention"

    def __init__(self, config: FuzzyTemporalTriangularConfig):
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
        return {
            "name": self.CONFIDENCE_ALGORITHM_NAME,
            "formula": self.CONFIDENCE_FORMULA,
            "description": (
                "Confidence is the max normalized combined score of the raw predicted cat after "
                "weight membership, triangular-selected dynamic sigma, optional penalties, and temporal prior."
            ),
            "parameters": {
                "temporal_weight": self.config.temporal_weight,
                "state_alpha": self.config.state_alpha,
                "decay_minutes": self.config.decay_minutes,
                "sigma_selection_method": "triangular_sigma_band_selection_in_pounds",
                "use_dynamic_sigma_in_prediction": self.config.use_dynamic_sigma_in_prediction,
                "use_overlap_penalty_in_prediction": self.config.use_overlap_penalty_in_prediction,
                "use_consistency_penalty_in_prediction": self.config.use_consistency_penalty_in_prediction,
                "overlap_penalty_strength": self.config.overlap_penalty_strength,
                "consistency_penalty_strength": self.config.consistency_penalty_strength,
                "base_sigma_lb": self.config.base_sigma_lb,
            },
        }

    def abstention_algorithm_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.ABSTENTION_ALGORITHM_NAME,
            "decision_rule": (
                "abstain if confidence < confidence_threshold OR "
                "predicted_overlap_for_abstention > overlap_abstention_threshold OR "
                "top1_top2_gap < score_gap_threshold"
            ),
            "abstain_label": self.config.abstain_label,
            "parameters": {
                "use_abstention": self.config.use_abstention,
                "confidence_threshold": self.config.confidence_threshold,
                "overlap_abstention_threshold": self.config.overlap_abstention_threshold,
                "score_gap_threshold": self.config.score_gap_threshold,
            },
        }

    def apply_overlap_penalty(self, scores, overlap_map):
        adjusted = {}
        for cat, score in scores.items():
            factor = 1.0 - self.config.overlap_penalty_strength * overlap_map.get(cat, 0.0)
            adjusted[cat] = score * max(factor, self.config.min_score_factor)
        return self.normalize_scores(adjusted)

    def apply_consistency_penalty(self, scores, consistency_map):
        adjusted = {}
        for cat, score in scores.items():
            consistency = consistency_map.get(cat, 1.0)
            if pd.isna(consistency):
                consistency = 1.0
            factor = 1.0 - self.config.consistency_penalty_strength * (1.0 - consistency)
            adjusted[cat] = score * max(factor, self.config.min_score_factor)
        return self.normalize_scores(adjusted)

    def apply_combined_abstention(
        self,
        raw_cat: str,
        confidence: float,
        combined: Dict[str, float],
        overlap_map: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Combined abstention strategy.

        Abstain if ANY of the following conditions is true:
        1. confidence < confidence_threshold
        2. overlap of raw predicted cat > overlap_abstention_threshold
        3. top1_top2_gap < score_gap_threshold
        """
        ordered = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
        top1 = ordered[0][1] if len(ordered) >= 1 else np.nan
        top2 = ordered[1][1] if len(ordered) >= 2 else np.nan
        gap = (top1 - top2) if len(ordered) >= 2 else 1.0
        predicted_overlap = overlap_map.get(raw_cat, 0.0) if overlap_map is not None else 0.0

        reasons = []
        if confidence < self.config.confidence_threshold:
            reasons.append("low_confidence")
        if predicted_overlap > self.config.overlap_abstention_threshold:
            reasons.append("high_overlap")
        if gap < self.config.score_gap_threshold:
            reasons.append("small_top1_top2_gap")

        abstained = self.config.use_abstention and bool(reasons)
        predicted_cat = self.config.abstain_label if abstained else raw_cat

        return {
            "raw_predicted_cat": raw_cat,
            "predicted_cat": predicted_cat,
            "abstained": abstained,
            "abstention_reason": ";".join(reasons) if reasons else "none",
            "top1_score": top1,
            "top2_score": top2,
            "top1_top2_gap": gap,
            "predicted_overlap_for_abstention": predicted_overlap,
        }

    def compute_weight_membership(self, row, catalog, sigma_map=None, overlap_map=None, consistency_map=None):
        measured_weight_kg = row["scale_weight"]
        raw_scores = {}
        for enroll_id, ref_weight_kg in catalog.items():
            if self.config.use_dynamic_sigma_in_prediction and sigma_map is not None:
                sigma_kg = sigma_map.get(enroll_id, self.config.base_sigma_kg)
            else:
                sigma_kg = self.config.base_sigma_kg

            raw_scores[enroll_id] = self.diagnostic.fuzzy_membership_with_sigma(
                measured_weight_kg,
                ref_weight_kg,
                sigma_kg,
            )

        scores = self.normalize_scores(raw_scores)
        if self.config.use_overlap_penalty_in_prediction and overlap_map is not None:
            scores = self.apply_overlap_penalty(scores, overlap_map)
        if self.config.use_consistency_penalty_in_prediction and consistency_map is not None:
            scores = self.apply_consistency_penalty(scores, consistency_map)
        return self.normalize_scores(scores)

    def predict_for_day(
        self,
        df_day,
        catalog,
        sigma_map=None,
        overlap_map=None,
        consistency_map=None,
        device_col="_device",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if df_day.empty:
            return pd.DataFrame(), {}
        df_day = df_day.sort_values("event_timestamp").copy()
        cats = list(catalog.keys())
        if not cats:
            return pd.DataFrame(), {}
        uniform = 1.0 / len(cats)
        latent_state = {cat: uniform for cat in cats}
        last_timestamp = None
        row_results = []
        cumulative_day_scores = {cat: 0.0 for cat in cats}
        for _, row in df_day.iterrows():
            current_membership = self.compute_weight_membership(row, catalog, sigma_map, overlap_map, consistency_map)
            if last_timestamp is None:
                gap_minutes = None
                decay = 0.0
            else:
                gap_minutes = (row["event_timestamp"] - last_timestamp).total_seconds() / 60.0
                decay = self.time_decay_factor(gap_minutes, self.config.decay_minutes)

            temporal_prior = {
                cat: decay * latent_state.get(cat, 0.0) + (1.0 - decay) * uniform
                for cat in cats
            }
            temporal_prior = self.normalize_scores(temporal_prior)

            combined = {
                cat: (1.0 - self.config.temporal_weight) * current_membership.get(cat, 0.0)
                + self.config.temporal_weight * temporal_prior.get(cat, 0.0)
                for cat in cats
            }
            combined = self.normalize_scores(combined)

            updated_state = {
                cat: self.config.state_alpha * combined.get(cat, 0.0)
                + (1.0 - self.config.state_alpha) * latent_state.get(cat, 0.0)
                for cat in cats
            }
            updated_state = self.normalize_scores(updated_state)
            
            # --- ABSTENTION EVALUATION ENGINE ---
            raw_cat = max(combined, key=combined.get)
            confidence = combined[raw_cat]
            abstention = self.apply_combined_abstention(raw_cat, confidence, combined, overlap_map)
            predicted_cat = abstention["predicted_cat"]

            predicted_weight_component = (1.0 - self.config.temporal_weight) * current_membership.get(raw_cat, 0.0)
            predicted_temporal_component = self.config.temporal_weight * temporal_prior.get(raw_cat, 0.0)
            for cat in cats:
                cumulative_day_scores[cat] += combined.get(cat, 0.0)
            row_result = {
                "user_id": row["user_id"],
                "device": row[device_col],
                "event_date": row["event_date"],
                "event_timestamp": row["event_timestamp"],
                "recorded_weight": row["scale_weight"],
                "true_label": row["true_label_cat_name"],
                "raw_predicted_cat": raw_cat,
                "predicted_cat": predicted_cat,
                "abstained": abstention["abstained"],
                "abstention_reason": abstention["abstention_reason"],
                "abstention_algorithm": self.ABSTENTION_ALGORITHM_NAME,
                "confidence": round(confidence, 6),
                "confidence_algorithm": self.CONFIDENCE_ALGORITHM_NAME,
                "confidence_formula": self.CONFIDENCE_FORMULA,
                "top1_score": round(abstention["top1_score"], 6),
                "top2_score": round(abstention["top2_score"], 6) if not pd.isna(abstention["top2_score"]) else np.nan,
                "top1_top2_gap": round(abstention["top1_top2_gap"], 6),
                "predicted_overlap_for_abstention": round(abstention["predicted_overlap_for_abstention"], 6),
                "confidence_threshold": self.config.confidence_threshold,
                "overlap_abstention_threshold": self.config.overlap_abstention_threshold,
                "score_gap_threshold": self.config.score_gap_threshold,
                "predicted_weight_component": round(predicted_weight_component, 6),
                "predicted_temporal_component": round(predicted_temporal_component, 6),
                "gap_minutes": gap_minutes if gap_minutes is not None else np.nan,
                "temporal_decay": round(decay, 6),
                "used_dynamic_sigma": self.config.use_dynamic_sigma_in_prediction,
                "used_overlap_penalty": self.config.use_overlap_penalty_in_prediction,
                "used_consistency_penalty": self.config.use_consistency_penalty_in_prediction,
                "sigma_selection_method": "triangular_sigma_band_selection_in_pounds",
            }
            for cat in cats:
                row_result[f"weight_score_{cat}"] = round(current_membership.get(cat, 0.0), 6)
                row_result[f"temporal_score_{cat}"] = round(temporal_prior.get(cat, 0.0), 6)
                row_result[f"combined_score_{cat}"] = round(combined.get(cat, 0.0), 6)
                if sigma_map is not None:
                    row_result[f"dynamic_sigma_kg_{cat}"] = round(sigma_map.get(cat, self.config.base_sigma_kg), 6)
                    row_result[f"dynamic_sigma_lb_{cat}"] = round(
                        sigma_map.get(cat, self.config.base_sigma_kg) * self.config.kg_to_lb,
                        6,
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
            "abstention_algorithm": self.ABSTENTION_ALGORITHM_NAME,
            "sigma_selection_method": "triangular_sigma_band_selection_in_pounds",
            "events_in_day": len(df_day),
            "abstained_events_in_day": int(row_result_df["abstained"].sum()) if not row_result_df.empty else 0,
        }
        for cat in cats:
            day_summary[f"day_total_score_{cat}"] = round(cumulative_day_scores.get(cat, 0.0), 6)
        return row_result_df, day_summary

    def run_user_prediction(self, df_user, catalog, inconsistency_df):
        sigma_map, consistency_map, overlap_map = self.diagnostic.build_prediction_artifacts_from_inconsistency(
            inconsistency_df,
            catalog,
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
                df_day,
                catalog,
                sigma_map,
                overlap_map,
                consistency_map,
                dev_col,
            )
            if not row_df.empty:
                all_day_rows.append(row_df)
            if day_summary:
                all_day_summaries.append(day_summary)

        result_df = pd.concat(all_day_rows, ignore_index=True) if all_day_rows else pd.DataFrame()
        daily_summary_df = pd.DataFrame(all_day_summaries)
        return result_df, daily_summary_df
