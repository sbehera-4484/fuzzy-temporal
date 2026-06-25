import math
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from config import FuzzyTemporalTriangularConfig


class FuzzyDiagnostic:
    """Label-free fuzzy diagnostic engine with triangular sigma-band selection."""
    def __init__(self, config: FuzzyTemporalTriangularConfig):
        self.config = config

    @staticmethod
    def normalize_scores(score_dict: Dict[str, float]) -> Dict[str, float]:
        total = sum(score_dict.values())
        if total == 0:
            n = len(score_dict)
            return {} if n == 0 else {k: 1.0 / n for k in score_dict}
        return {k: v / total for k, v in score_dict.items()}

    def safe_sigma_kg(self, value_kg) -> float:
        if value_kg is None or pd.isna(value_kg) or value_kg <= 0:
            return self.config.base_sigma_kg
        return min(max(value_kg, self.config.min_dynamic_sigma_kg), self.config.max_dynamic_sigma_kg)

    def fuzzy_membership_with_sigma(self, measured_weight_kg: float, reference_weight_kg: float, sigma_kg: float) -> float:
        sigma_kg = self.safe_sigma_kg(sigma_kg)
        return math.exp(-0.5 * ((measured_weight_kg - reference_weight_kg) / sigma_kg) ** 2)

    @staticmethod
    def triangular_membership(x: float, left: float, center: float, right) -> float:
        if right is None or x <= left or x >= right:
            return 0.0
        if x == center:
            return 1.0
        if left < x < center:
            return (x - left) / (center - left)
        if center < x < right:
            return (right - x) / (right - center)
        return 0.0

    @staticmethod
    def right_shoulder_membership(x: float, left: float, center: float) -> float:
        if x <= left:
            return 0.0
        if x >= center:
            return 1.0
        return (x - left) / (center - left)

    def sigma_band_memberships_lb(self, raw_sigma_lb: float) -> Dict[str, float]:
        memberships = {}
        for band_name, params in self.config.sigma_bands_lb.items():
            if params["right"] is None:
                memberships[band_name] = self.right_shoulder_membership(raw_sigma_lb, params["left"], params["center"])
            else:
                memberships[band_name] = self.triangular_membership(raw_sigma_lb, params["left"], params["center"], params["right"])
        return memberships

    def select_sigma_using_triangular_bands(self, raw_sigma_kg) -> Tuple[float, float, str, Dict[str, float]]:
        if raw_sigma_kg is None or pd.isna(raw_sigma_kg) or raw_sigma_kg <= 0:
            raw_sigma_kg = self.config.base_sigma_kg
        raw_sigma_lb = raw_sigma_kg * self.config.kg_to_lb
        raw_sigma_lb_for_selection = max(raw_sigma_lb, 0.5)
        memberships = self.sigma_band_memberships_lb(raw_sigma_lb_for_selection)
        total = sum(memberships.values())
        if total == 0:
            selected_sigma_lb = self.config.base_sigma_lb
            dominant_band = "fallback"
        else:
            selected_sigma_lb = sum(
                mu * self.config.sigma_bands_lb[band]["representative"] for band, mu in memberships.items()
            ) / total
            dominant_band = max(memberships, key=memberships.get)
        selected_sigma_kg = self.safe_sigma_kg(selected_sigma_lb * self.config.lb_to_kg)
        return selected_sigma_kg, selected_sigma_kg * self.config.kg_to_lb, dominant_band, memberships

    def estimate_dynamic_sigma_fuzzy_with_triangular_selection(self, df_user: pd.DataFrame, catalog: Dict[str, float]):
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
                    raw_scores = {
                        enroll_id: self.fuzzy_membership_with_sigma(
                            measured_weight_kg, ref_weight_kg, sigma_map.get(enroll_id, self.config.base_sigma_kg)
                        )
                        for enroll_id, ref_weight_kg in catalog.items()
                    }
                    probs = self.normalize_scores(raw_scores)
                    mu = probs.get(target_id, 0.0)
                    numerator += mu * ((measured_weight_kg - target_ref_weight_kg) ** 2)
                    denominator += mu
                updated_raw_sigma_map[target_id] = math.sqrt(numerator / denominator) if denominator > 0 else self.config.base_sigma_kg
            sigma_map = {
                enroll_id: self.select_sigma_using_triangular_bands(raw_sigma_kg)[0]
                for enroll_id, raw_sigma_kg in updated_raw_sigma_map.items()
            }
            raw_sigma_map = updated_raw_sigma_map
        sigma_debug_map = {}
        for enroll_id, raw_sigma_kg in raw_sigma_map.items():
            selected_kg, selected_lb, dominant_band, memberships = self.select_sigma_using_triangular_bands(raw_sigma_kg)
            sigma_map[enroll_id] = selected_kg
            sigma_debug_map[enroll_id] = {
                "raw_sigma_kg": raw_sigma_kg,
                "raw_sigma_lb": raw_sigma_kg * self.config.kg_to_lb,
                "selected_sigma_kg": selected_kg,
                "selected_sigma_lb": selected_lb,
                "dominant_sigma_band": dominant_band,
                **memberships,
            }
        return sigma_map, sigma_debug_map

    def fuzzy_overlap_score(self, ref_weight_1_kg, ref_weight_2_kg, sigma_1_kg, sigma_2_kg) -> float:
        pair_sigma_kg = self.safe_sigma_kg((sigma_1_kg + sigma_2_kg) / 2.0)
        diff_kg = abs(ref_weight_1_kg - ref_weight_2_kg)
        return math.exp(-0.5 * (diff_kg / pair_sigma_kg) ** 2)

    def fuzzy_soft_consistency_score(self, df_user, catalog, target_id, reference_weight_kg, sigma_map) -> float:
        numerator = 0.0
        denominator = 0.0
        usable_df = df_user.dropna(subset=["scale_weight"]).copy()
        for _, row in usable_df.iterrows():
            measured_weight_kg = row["scale_weight"]
            raw_scores = {
                enroll_id: self.fuzzy_membership_with_sigma(measured_weight_kg, ref_weight_kg, sigma_map.get(enroll_id, self.config.base_sigma_kg))
                for enroll_id, ref_weight_kg in catalog.items()
            }
            probs = self.normalize_scores(raw_scores)
            mu = probs.get(target_id, 0.0)
            own_fit = self.fuzzy_membership_with_sigma(measured_weight_kg, reference_weight_kg, sigma_map.get(target_id, self.config.base_sigma_kg))
            numerator += mu * own_fit
            denominator += mu
        return np.nan if denominator == 0 else numerator / denominator

    def analyze_cat_inconsistency_fuzzy(self, df_user: pd.DataFrame, catalog: Dict[str, float]) -> pd.DataFrame:
        records = []
        if df_user.empty or len(catalog) == 0:
            return pd.DataFrame()
        sigma_map, sigma_debug_map = self.estimate_dynamic_sigma_fuzzy_with_triangular_selection(df_user, catalog)
        all_weights = df_user["scale_weight"].dropna().tolist()
        global_scale_mean_kg = float(np.mean(all_weights)) if all_weights else np.nan
        global_scale_std_kg = float(np.std(all_weights, ddof=1)) if len(all_weights) > 1 else (0.0 if all_weights else np.nan)
        global_scale_min_kg = float(np.min(all_weights)) if all_weights else np.nan
        global_scale_max_kg = float(np.max(all_weights)) if all_weights else np.nan
        for enroll_id, reference_weight_kg in catalog.items():
            sigma_info = sigma_debug_map.get(enroll_id, {})
            selected_sigma_kg = sigma_map.get(enroll_id, self.config.base_sigma_kg)
            consistency_score = self.fuzzy_soft_consistency_score(df_user, catalog, enroll_id, reference_weight_kg, sigma_map)
            if pd.isna(consistency_score):
                variation_risk = "unknown"
            elif consistency_score < self.config.low_consistency_score:
                variation_risk = "high_variation"
            elif consistency_score < self.config.medium_consistency_score:
                variation_risk = "medium_variation"
            else:
                variation_risk = "stable"
            row = {
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
                "dominant_sigma_band": sigma_info.get("dominant_sigma_band", None),
                "fuzzy_consistency_score": round(consistency_score, 6) if not pd.isna(consistency_score) else np.nan,
                "variation_risk": variation_risk,
                "global_scale_mean_kg": round(global_scale_mean_kg, 6) if not pd.isna(global_scale_mean_kg) else np.nan,
                "global_scale_std_kg": round(global_scale_std_kg, 6) if not pd.isna(global_scale_std_kg) else np.nan,
                "global_scale_min_kg": round(global_scale_min_kg, 6) if not pd.isna(global_scale_min_kg) else np.nan,
                "global_scale_max_kg": round(global_scale_max_kg, 6) if not pd.isna(global_scale_max_kg) else np.nan,
                "total_measurements_used": len(all_weights),
            }
            for band_name in self.config.sigma_bands_lb.keys():
                row[f"mu_{band_name}"] = round(sigma_info.get(band_name, 0.0), 6)
            records.append(row)
        ids = list(catalog.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                ref1_kg, ref2_kg = catalog[id1], catalog[id2]
                sigma1_kg, sigma2_kg = sigma_map.get(id1, self.config.base_sigma_kg), sigma_map.get(id2, self.config.base_sigma_kg)
                pair_sigma_kg = self.safe_sigma_kg((sigma1_kg + sigma2_kg) / 2.0)
                reference_diff_kg = abs(ref1_kg - ref2_kg)
                overlap_score = self.fuzzy_overlap_score(ref1_kg, ref2_kg, sigma1_kg, sigma2_kg)
                overlap_risk = "high_overlap" if overlap_score >= self.config.high_overlap_score else "medium_overlap" if overlap_score >= self.config.medium_overlap_score else "low_overlap"
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

    def build_prediction_artifacts_from_inconsistency(self, inconsistency_df: pd.DataFrame, catalog: Dict[str, float]):
        sigma_map = {enroll_id: self.config.base_sigma_kg for enroll_id in catalog.keys()}
        consistency_map = {enroll_id: 1.0 for enroll_id in catalog.keys()}
        overlap_map = {enroll_id: 0.0 for enroll_id in catalog.keys()}
        if inconsistency_df is None or inconsistency_df.empty:
            return sigma_map, consistency_map, overlap_map
        for _, row in inconsistency_df[inconsistency_df["type"] == "fuzzy_profile_weight_consistency"].iterrows():
            enroll_id = row.get("enroll_id")
            if enroll_id not in catalog:
                continue
            sigma_map[enroll_id] = self.safe_sigma_kg(row.get("selected_dynamic_sigma_kg", self.config.base_sigma_kg))
            consistency = row.get("fuzzy_consistency_score", 1.0)
            consistency_map[enroll_id] = 1.0 if pd.isna(consistency) else float(consistency)
        for _, row in inconsistency_df[inconsistency_df["type"] == "fuzzy_overlap"].iterrows():
            id1, id2 = row.get("enroll_id_1"), row.get("enroll_id_2")
            overlap_score = row.get("fuzzy_overlap_score", 0.0)
            overlap_score = 0.0 if pd.isna(overlap_score) else float(overlap_score)
            if id1 in overlap_map:
                overlap_map[id1] = max(overlap_map[id1], overlap_score)
            if id2 in overlap_map:
                overlap_map[id2] = max(overlap_map[id2], overlap_score)
        return sigma_map, consistency_map, overlap_map
