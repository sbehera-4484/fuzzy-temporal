import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config import FuzzyTemporalConfig


class FuzzyDiagnostic:
    """
    Label-free fuzzy diagnostic engine.

    It estimates:
    1. continuous dynamic sigma per enrolled cat
    2. fuzzy profile-weight consistency
    3. fuzzy overlap between enrolled cats

    This class does not load files and does not use true labels for prediction.
    """

    def __init__(self, config: FuzzyTemporalConfig):
        self.config = config

    @staticmethod
    def normalize_scores(score_dict: Dict[str, float]) -> Dict[str, float]:
        total = sum(score_dict.values())
        if total == 0:
            n = len(score_dict)
            if n == 0:
                return {}
            return {k: 1.0 / n for k in score_dict}
        return {k: v / total for k, v in score_dict.items()}

    def safe_sigma_kg(self, value_kg) -> float:
        """Return bounded sigma in kg. Fallback sigma is configured in pounds."""
        if value_kg is None or pd.isna(value_kg) or value_kg <= 0:
            return self.config.base_sigma_kg

        return min(
            max(value_kg, self.config.min_dynamic_sigma_kg),
            self.config.max_dynamic_sigma_kg,
        )

    def fuzzy_membership_with_sigma(
        self,
        measured_weight_kg: float,
        reference_weight_kg: float,
        sigma_kg: float,
    ) -> float:
        """Gaussian fuzzy membership."""
        sigma_kg = self.safe_sigma_kg(sigma_kg)
        exponent = -0.5 * ((measured_weight_kg - reference_weight_kg) / sigma_kg) ** 2
        return math.exp(exponent)

    def estimate_dynamic_sigma_fuzzy_continuous(
        self,
        df_user: pd.DataFrame,
        catalog: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Estimate per-cat dynamic sigma without true labels.

        Uses fuzzy soft assignment and weighted residual variance.
        """
        sigma_map = {enroll_id: self.config.base_sigma_kg for enroll_id in catalog.keys()}
        usable_df = df_user.dropna(subset=["scale_weight"]).copy()

        if usable_df.empty or len(catalog) == 0:
            return sigma_map, {}

        raw_sigma_map = {}

        for _ in range(self.config.sigma_iterations):
            updated_raw_sigma_map = {}

            for target_id, target_ref_weight_kg in catalog.items():
                numerator = 0.0
                denominator = 0.0

                for _, row in usable_df.iterrows():
                    measured_weight_kg = row["scale_weight"]
                    raw_scores = {}

                    for enroll_id, ref_weight_kg in catalog.items():
                        raw_scores[enroll_id] = self.fuzzy_membership_with_sigma(
                            measured_weight_kg=measured_weight_kg,
                            reference_weight_kg=ref_weight_kg,
                            sigma_kg=sigma_map.get(enroll_id, self.config.base_sigma_kg),
                        )

                    probs = self.normalize_scores(raw_scores)
                    mu = probs.get(target_id, 0.0)

                    numerator += mu * ((measured_weight_kg - target_ref_weight_kg) ** 2)
                    denominator += mu

                raw_sigma_kg = math.sqrt(numerator / denominator) if denominator > 0 else self.config.base_sigma_kg
                updated_raw_sigma_map[target_id] = raw_sigma_kg

            sigma_map = {
                enroll_id: self.safe_sigma_kg(raw_sigma_kg)
                for enroll_id, raw_sigma_kg in updated_raw_sigma_map.items()
            }
            raw_sigma_map = updated_raw_sigma_map

        sigma_debug_map = {}
        for enroll_id, raw_sigma_kg in raw_sigma_map.items():
            selected_sigma_kg = sigma_map.get(enroll_id, self.config.base_sigma_kg)
            sigma_debug_map[enroll_id] = {
                "raw_sigma_kg": raw_sigma_kg,
                "raw_sigma_lb": raw_sigma_kg * self.config.kg_to_lb,
                "selected_sigma_kg": selected_sigma_kg,
                "selected_sigma_lb": selected_sigma_kg * self.config.kg_to_lb,
            }

        return sigma_map, sigma_debug_map

    def fuzzy_overlap_score(
        self,
        ref_weight_1_kg: float,
        ref_weight_2_kg: float,
        sigma_1_kg: float,
        sigma_2_kg: float,
    ) -> float:
        """Compute fuzzy overlap between two enrolled cats."""
        pair_sigma_kg = self.safe_sigma_kg((sigma_1_kg + sigma_2_kg) / 2.0)
        diff_kg = abs(ref_weight_1_kg - ref_weight_2_kg)
        return math.exp(-0.5 * (diff_kg / pair_sigma_kg) ** 2)

    def fuzzy_soft_consistency_score(
        self,
        df_user: pd.DataFrame,
        catalog: Dict[str, float],
        target_id: str,
        reference_weight_kg: float,
        sigma_map: Dict[str, float],
    ) -> float:
        """Compute label-free profile consistency for one enrolled cat."""
        numerator = 0.0
        denominator = 0.0
        usable_df = df_user.dropna(subset=["scale_weight"]).copy()

        for _, row in usable_df.iterrows():
            measured_weight_kg = row["scale_weight"]
            raw_scores = {}

            for enroll_id, ref_weight_kg in catalog.items():
                raw_scores[enroll_id] = self.fuzzy_membership_with_sigma(
                    measured_weight_kg=measured_weight_kg,
                    reference_weight_kg=ref_weight_kg,
                    sigma_kg=sigma_map.get(enroll_id, self.config.base_sigma_kg),
                )

            probs = self.normalize_scores(raw_scores)
            mu = probs.get(target_id, 0.0)

            own_fit = self.fuzzy_membership_with_sigma(
                measured_weight_kg=measured_weight_kg,
                reference_weight_kg=reference_weight_kg,
                sigma_kg=sigma_map.get(target_id, self.config.base_sigma_kg),
            )

            numerator += mu * own_fit
            denominator += mu

        if denominator == 0:
            return np.nan
        return numerator / denominator

    def analyze_cat_inconsistency_fuzzy(
        self,
        df_user: pd.DataFrame,
        catalog: Dict[str, float],
    ) -> pd.DataFrame:
        """Generate fuzzy inconsistency dataframe."""
        records = []

        if df_user.empty or len(catalog) == 0:
            return pd.DataFrame()

        sigma_map, sigma_debug_map = self.estimate_dynamic_sigma_fuzzy_continuous(
            df_user=df_user,
            catalog=catalog,
        )

        all_weights = df_user["scale_weight"].dropna().tolist()

        if len(all_weights) > 0:
            global_scale_mean_kg = float(np.mean(all_weights))
            global_scale_std_kg = float(np.std(all_weights, ddof=1)) if len(all_weights) > 1 else 0.0
            global_scale_min_kg = float(np.min(all_weights))
            global_scale_max_kg = float(np.max(all_weights))
        else:
            global_scale_mean_kg = np.nan
            global_scale_std_kg = np.nan
            global_scale_min_kg = np.nan
            global_scale_max_kg = np.nan

        for enroll_id, reference_weight_kg in catalog.items():
            sigma_info = sigma_debug_map.get(enroll_id, {})
            selected_sigma_kg = sigma_map.get(enroll_id, self.config.base_sigma_kg)

            consistency_score = self.fuzzy_soft_consistency_score(
                df_user=df_user,
                catalog=catalog,
                target_id=enroll_id,
                reference_weight_kg=reference_weight_kg,
                sigma_map=sigma_map,
            )

            if pd.isna(consistency_score):
                variation_risk = "unknown"
            elif consistency_score < self.config.low_consistency_score:
                variation_risk = "high_variation"
            elif consistency_score < self.config.medium_consistency_score:
                variation_risk = "medium_variation"
            else:
                variation_risk = "stable"

            records.append({
                "type": "fuzzy_profile_weight_consistency",
                "enroll_id": enroll_id,
                "reference_weight_kg": round(reference_weight_kg, 6),
                "reference_weight_lb": round(reference_weight_kg * self.config.kg_to_lb, 6),
                "fallback_sigma_lb": round(self.config.base_sigma_lb, 6),
                "fallback_sigma_kg": round(self.config.base_sigma_kg, 6),
                "raw_sigma_kg": round(sigma_info.get("raw_sigma_kg", np.nan), 6),
                "raw_sigma_lb": round(sigma_info.get("raw_sigma_lb", np.nan), 6),
                "selected_dynamic_sigma_kg": round(selected_sigma_kg, 6),
                "selected_dynamic_sigma_lb": round(selected_sigma_kg * self.config.kg_to_lb, 6),
                "sigma_selection_method": "continuous_dynamic_sigma_no_triangular_bands",
                "fuzzy_consistency_score": round(consistency_score, 6) if not pd.isna(consistency_score) else np.nan,
                "variation_risk": variation_risk,
                "global_scale_mean_kg": round(global_scale_mean_kg, 6) if not pd.isna(global_scale_mean_kg) else np.nan,
                "global_scale_std_kg": round(global_scale_std_kg, 6) if not pd.isna(global_scale_std_kg) else np.nan,
                "global_scale_min_kg": round(global_scale_min_kg, 6) if not pd.isna(global_scale_min_kg) else np.nan,
                "global_scale_max_kg": round(global_scale_max_kg, 6) if not pd.isna(global_scale_max_kg) else np.nan,
                "total_measurements_used": len(all_weights),
            })

        ids = list(catalog.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1 = ids[i]
                id2 = ids[j]
                ref1_kg = catalog[id1]
                ref2_kg = catalog[id2]

                sigma1_kg = sigma_map.get(id1, self.config.base_sigma_kg)
                sigma2_kg = sigma_map.get(id2, self.config.base_sigma_kg)
                pair_sigma_kg = self.safe_sigma_kg((sigma1_kg + sigma2_kg) / 2.0)
                reference_diff_kg = abs(ref1_kg - ref2_kg)
                overlap_score = self.fuzzy_overlap_score(ref1_kg, ref2_kg, sigma1_kg, sigma2_kg)

                if overlap_score >= self.config.high_overlap_score:
                    overlap_risk = "high_overlap"
                elif overlap_score >= self.config.medium_overlap_score:
                    overlap_risk = "medium_overlap"
                else:
                    overlap_risk = "low_overlap"

                records.append({
                    "type": "fuzzy_overlap",
                    "enroll_id_1": id1,
                    "enroll_id_2": id2,
                    "reference_weight_1_kg": round(ref1_kg, 6),
                    "reference_weight_2_kg": round(ref2_kg, 6),
                    "reference_weight_1_lb": round(ref1_kg * self.config.kg_to_lb, 6),
                    "reference_weight_2_lb": round(ref2_kg * self.config.kg_to_lb, 6),
                    "reference_diff_kg": round(reference_diff_kg, 6),
                    "reference_diff_lb": round(reference_diff_kg * self.config.kg_to_lb, 6),
                    "selected_dynamic_sigma_1_kg": round(sigma1_kg, 6),
                    "selected_dynamic_sigma_2_kg": round(sigma2_kg, 6),
                    "selected_dynamic_sigma_1_lb": round(sigma1_kg * self.config.kg_to_lb, 6),
                    "selected_dynamic_sigma_2_lb": round(sigma2_kg * self.config.kg_to_lb, 6),
                    "pair_sigma_kg": round(pair_sigma_kg, 6),
                    "pair_sigma_lb": round(pair_sigma_kg * self.config.kg_to_lb, 6),
                    "fuzzy_overlap_score": round(overlap_score, 6),
                    "overlap_risk": overlap_risk,
                })

        return pd.DataFrame(records)

    def build_prediction_artifacts_from_inconsistency(
        self,
        inconsistency_df: pd.DataFrame,
        catalog: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Convert inconsistency dataframe into prediction-time maps.

        Returns:
            sigma_map, consistency_map, overlap_map
        """
        sigma_map = {enroll_id: self.config.base_sigma_kg for enroll_id in catalog.keys()}
        consistency_map = {enroll_id: 1.0 for enroll_id in catalog.keys()}
        overlap_map = {enroll_id: 0.0 for enroll_id in catalog.keys()}

        if inconsistency_df is None or inconsistency_df.empty:
            return sigma_map, consistency_map, overlap_map

        consistency_rows = inconsistency_df[
            inconsistency_df["type"] == "fuzzy_profile_weight_consistency"
        ]

        for _, row in consistency_rows.iterrows():
            enroll_id = row.get("enroll_id")
            if enroll_id not in catalog:
                continue

            sigma_kg = row.get("selected_dynamic_sigma_kg", self.config.base_sigma_kg)
            consistency = row.get("fuzzy_consistency_score", 1.0)

            sigma_map[enroll_id] = self.safe_sigma_kg(sigma_kg)
            consistency_map[enroll_id] = 1.0 if pd.isna(consistency) else float(consistency)

        overlap_rows = inconsistency_df[inconsistency_df["type"] == "fuzzy_overlap"]

        for _, row in overlap_rows.iterrows():
            id1 = row.get("enroll_id_1")
            id2 = row.get("enroll_id_2")
            overlap_score = row.get("fuzzy_overlap_score", 0.0)
            overlap_score = 0.0 if pd.isna(overlap_score) else float(overlap_score)

            if id1 in overlap_map:
                overlap_map[id1] = max(overlap_map[id1], overlap_score)
            if id2 in overlap_map:
                overlap_map[id2] = max(overlap_map[id2], overlap_score)

        return sigma_map, consistency_map, overlap_map
