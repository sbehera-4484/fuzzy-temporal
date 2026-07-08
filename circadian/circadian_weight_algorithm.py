#!/usr/bin/env python3
"""
Single-file pandas-based CircadianWeightAlgorithm with confidence + abstention logic.

Key API design requested:
    1. datareadingFunction(csv_path) reads CSV and returns prepared DataFrame
    2. run(df) takes DataFrame as input, not CSV path
    3. run(df) returns result DataFrame in memory
    4. save_results(result_df, output_csv) saves result DataFrame as CSV

Input CSV schema expected, for example 1000156.csv:
    serial,user_id,total_pets,from_pet_id,to_pet_id,profile_weight,
    visit_date,visit_time,scale_weight,true_label_pet_id,true_label_cat_name

Dependency:
    pandas

Install if needed:
    pip install pandas

Example:
    algo = CircadianWeightAlgorithm()
    df = algo.datareadingFunction("1000156.csv")
    result_df = algo.run(df)
    algo.save_results(result_df, "1000156_circadian_results.csv")
    algo.print_summary(result_df)
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Optional

import pandas as pd


class CircadianWeightAlgorithm:
    """
    Single-class pandas implementation of Circadian Weight Algorithm.

    Includes:
        - CSV reading and preprocessing using pandas
        - Pet profile creation from DataFrame
        - Weight likelihood
        - Circadian hour prior
        - Confidence calculation
        - Abstention logic
        - DataFrame output
        - CSV saving helper

    Scoring formula:
        combined_score = weight_likelihood * (1 + effective_boost * (hour_prior / baseline - 1))

    Confidence:
        posterior = combined_score / sum(all_pet_combined_scores)
        confidence = top posterior
        confidence_margin = top posterior - second-best posterior

    Abstention reasons:
        LOW_CONFIDENCE
        LOW_MARGIN
        HIGH_WEIGHT_AMBIGUITY
        WEIGHT_OUTLIER
        LOW_LIKELIHOOD
    """

    def __init__(
        self,
        sigma: float = 0.4,
        hour_boost: float = 0.5,
        close_weight_threshold: float = 1.0,
        reassignment_weight: float = 3.0,
        ema_alpha: float = 0.15,
        max_recent_weights: int = 60,
        warm_start_with_true_labels: bool = True,
        online_learning: bool = True,
        never_visited_penalty_value: float = 0.97,
        enable_abstention: bool = True,
        min_confidence: float = 0.60,
        min_confidence_margin: float = 0.15,
        high_ambiguity_threshold: float = 0.50,
        max_weight_zscore: float = 3.0,
        min_winner_likelihood: float = 0.05,
    ) -> None:
        self.sigma = sigma
        self.hour_boost = hour_boost
        self.close_weight_threshold = close_weight_threshold
        self.reassignment_weight = reassignment_weight
        self.ema_alpha = ema_alpha
        self.max_recent_weights = max_recent_weights
        self.warm_start_with_true_labels = warm_start_with_true_labels
        self.online_learning = online_learning
        self.never_visited_penalty_value = never_visited_penalty_value

        self.enable_abstention = enable_abstention
        self.min_confidence = min_confidence
        self.min_confidence_margin = min_confidence_margin
        self.high_ambiguity_threshold = high_ambiguity_threshold
        self.max_weight_zscore = max_weight_zscore
        self.min_winner_likelihood = min_winner_likelihood

        self.pet_profiles: Dict[str, Dict[str, Any]] = {}
        self.assigned_visit_counts: Dict[str, int] = {}

    # ==================================================================
    # 1. CSV READING + DATAFRAME PREPARATION
    # ==================================================================
    def datareadingFunction(self, csv_path: str = "1000156.csv") -> pd.DataFrame:
        """
        Read CSV using pandas and return a prepared DataFrame.

        This function reads from CSV only. The algorithm run() function does not
        read CSV; it takes this returned DataFrame as input.
        """
        df = pd.read_csv(csv_path)
        return self.prepare_dataframe(df)

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare already-loaded DataFrame.

        This is useful when you already have df = pd.read_csv(...)
        and want to pass it through the same cleaning logic.
        """
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        required_columns = [
            "serial",
            "user_id",
            "total_pets",
            "profile_weight",
            "visit_date",
            "visit_time",
            "scale_weight",
            "true_label_pet_id",
            "true_label_cat_name",
        ]
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df["profile_weight"] = pd.to_numeric(df["profile_weight"], errors="coerce")
        df["scale_weight"] = pd.to_numeric(df["scale_weight"], errors="coerce")
        df["total_pets"] = pd.to_numeric(df["total_pets"], errors="coerce")

        df["visit_datetime"] = pd.to_datetime(
            df["visit_date"].astype(str).str.strip()
            + " "
            + df["visit_time"].astype(str).str.strip(),
            errors="coerce",
        )

        df = df.dropna(subset=["visit_datetime", "scale_weight"]).copy()
        df["visit_hour"] = df["visit_datetime"].dt.hour

        # Microsecond timestamp, similar to original implementation convention
        df["timestamp"] = (df["visit_datetime"].astype("int64") // 1000).astype("int64")

        df["true_label_pet_id"] = df["true_label_pet_id"].fillna("").astype(str).str.strip()
        df["true_label_cat_name"] = df["true_label_cat_name"].fillna("").astype(str).str.strip()

        if "from_pet_id" not in df.columns:
            df["from_pet_id"] = ""
        if "to_pet_id" not in df.columns:
            df["to_pet_id"] = ""

        df = df.sort_values("visit_datetime").reset_index(drop=True)
        return df

    # ==================================================================
    # 2. BUILD PET PROFILES FROM DATAFRAME
    # ==================================================================
    def build_pet_profiles(self, df: pd.DataFrame) -> None:
        """
        Build pet profiles from true_label_pet_id and profile_weight.

        For each pet:
            - weight_anchor = median(profile_weight)
            - if profile_weight missing, fallback to median(scale_weight)
            - circadian histogram initialized as 24 bins
        """
        labelled_df = df[df["true_label_pet_id"].astype(str).str.len() > 0].copy()

        self.pet_profiles = {}
        self.assigned_visit_counts = {}

        for pet_id, g in labelled_df.groupby("true_label_pet_id"):
            cat_names = g["true_label_cat_name"].dropna().astype(str).str.strip()
            pet_name = cat_names.mode().iloc[0] if not cat_names.empty else pet_id

            profile_weights = g["profile_weight"].dropna()
            scale_weights = g["scale_weight"].dropna()

            if len(profile_weights) > 0:
                weight_anchor = float(profile_weights.median())
            elif len(scale_weights) > 0:
                weight_anchor = float(scale_weights.median())
            else:
                weight_anchor = 0.0

            self.pet_profiles[pet_id] = {
                "pet_id": pet_id,
                "pet_name": pet_name,
                "is_active": True,
                "weight_anchor": weight_anchor,
                "hour_counts": [0.0] * 24,
                "total_visits": 0,
                "ema_weight": weight_anchor,
                "visit_count": 0,
                "weights": [],
            }
            self.assigned_visit_counts[pet_id] = 0

        # Warm-start circadian histogram using labelled data.
        # Disable by setting warm_start_with_true_labels=False.
        if self.warm_start_with_true_labels:
            for _, row in labelled_df.iterrows():
                pet_id = row["true_label_pet_id"]
                hour = int(row["visit_hour"])
                self.record_hour(pet_id, hour, weight=self.reassignment_weight)

    # ==================================================================
    # 3. PROFILE UPDATE FUNCTIONS
    # ==================================================================
    def record_hour(self, pet_id: str, hour: int, weight: float = 1.0) -> None:
        profile = self.pet_profiles[pet_id]
        profile["hour_counts"][hour % 24] += weight
        profile["total_visits"] += 1

    def record_visit_weight(self, pet_id: str, weight_lbs: float) -> None:
        profile = self.pet_profiles[pet_id]

        if profile["visit_count"] == 0:
            profile["ema_weight"] = weight_lbs
        else:
            profile["ema_weight"] = (
                self.ema_alpha * weight_lbs
                + (1.0 - self.ema_alpha) * profile["ema_weight"]
            )

        profile["visit_count"] += 1
        profile["weights"].append(weight_lbs)

        if len(profile["weights"]) > self.max_recent_weights:
            profile["weights"] = profile["weights"][-self.max_recent_weights:]

    def effective_anchor(self, pet_id: str) -> float:
        profile = self.pet_profiles[pet_id]
        if profile["visit_count"] >= 3:
            return float(profile["ema_weight"])
        return float(profile["weight_anchor"])

    def weight_std(self, pet_id: str) -> float:
        profile = self.pet_profiles[pet_id]
        weights = profile["weights"]

        if len(weights) < 3:
            return max(self.sigma, 0.15)

        std = float(pd.Series(weights, dtype="float64").std(ddof=0))
        return max(std, 0.15)

    def hour_probability(self, pet_id: str, hour: int) -> float:
        profile = self.pet_profiles[pet_id]
        hour_counts = profile["hour_counts"]

        if profile["total_visits"] == 0:
            return 1.0 / 24.0

        h = hour % 24
        smoothed = [c + 0.5 for c in hour_counts]
        total = sum(smoothed)

        raw_prob = smoothed[h] / total
        neighbor_avg_count = (
            smoothed[(h - 1) % 24]
            + smoothed[h]
            + smoothed[(h + 1) % 24]
        ) / 3.0
        neighbor_prob = neighbor_avg_count / total

        return float((raw_prob + neighbor_prob) / 2.0)

    # ==================================================================
    # 4. SCORING FUNCTIONS
    # ==================================================================
    def weight_likelihood(self, weight_lbs: float, pet_id: str) -> float:
        anchor = self.effective_anchor(pet_id)
        std = max(self.weight_std(pet_id), 0.2)
        diff = weight_lbs - anchor
        return float(math.exp(-0.5 * (diff / std) ** 2))

    def weight_zscore(self, weight_lbs: float, pet_id: str) -> float:
        anchor = self.effective_anchor(pet_id)
        std = max(self.weight_std(pet_id), 0.2)
        return float(abs(weight_lbs - anchor) / std)

    def weight_ambiguity(self, weight_lbs: float) -> float:
        close_count = 0

        for pet_id, profile in self.pet_profiles.items():
            if not profile.get("is_active", True):
                continue
            anchor = self.effective_anchor(pet_id)
            if abs(weight_lbs - anchor) <= self.close_weight_threshold:
                close_count += 1

        if close_count <= 1:
            return 0.0
        return float(min((close_count - 1) / 4.0, 1.0))

    def never_visited_penalty(self, pet_id: str) -> float:
        if self.assigned_visit_counts.get(pet_id, 0) == 0:
            return self.never_visited_penalty_value
        return 1.0

    # ==================================================================
    # 5. CONFIDENCE + ABSTENTION LOGIC
    # ==================================================================
    def calculate_confidence_metrics(self, scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not scores:
            return {
                "confidence": 0.0,
                "second_best_confidence": 0.0,
                "confidence_margin": 0.0,
                "score_entropy": 0.0,
                "normalized_entropy": 0.0,
            }

        ranked = sorted(scores, key=lambda x: x["posterior"], reverse=True)
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None

        confidence = float(top["posterior"])
        second_confidence = float(second["posterior"]) if second else 0.0
        margin = confidence - second_confidence

        entropy = 0.0
        for s in ranked:
            p = max(float(s["posterior"]), 1e-12)
            entropy += -p * math.log(p)

        max_entropy = math.log(len(ranked)) if len(ranked) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return {
            "confidence": confidence,
            "second_best_confidence": second_confidence,
            "confidence_margin": margin,
            "score_entropy": entropy,
            "normalized_entropy": normalized_entropy,
        }

    def apply_abstention_logic(
        self,
        winner: Dict[str, Any],
        second_best: Optional[Dict[str, Any]],
        confidence_metrics: Dict[str, Any],
        ambiguity: float,
        weight_lbs: float,
    ) -> Dict[str, Any]:
        if not self.enable_abstention:
            return {
                "abstained": False,
                "abstain_reason": "NOT_ENABLED",
                "final_pet_id": winner["pet_id"],
                "final_pet_name": winner["pet_name"],
            }

        reasons: List[str] = []

        confidence = float(confidence_metrics["confidence"])
        margin = float(confidence_metrics["confidence_margin"])
        winner_likelihood = float(winner["likelihood"])
        winner_zscore = self.weight_zscore(weight_lbs, winner["pet_id"])

        if confidence < self.min_confidence:
            reasons.append("LOW_CONFIDENCE")
        if margin < self.min_confidence_margin:
            reasons.append("LOW_MARGIN")
        if ambiguity >= self.high_ambiguity_threshold:
            reasons.append("HIGH_WEIGHT_AMBIGUITY")
        if winner_zscore > self.max_weight_zscore:
            reasons.append("WEIGHT_OUTLIER")
        if winner_likelihood < self.min_winner_likelihood:
            reasons.append("LOW_LIKELIHOOD")

        if reasons:
            return {
                "abstained": True,
                "abstain_reason": "+".join(reasons),
                "final_pet_id": "ABSTAIN",
                "final_pet_name": "ABSTAIN",
            }

        return {
            "abstained": False,
            "abstain_reason": "",
            "final_pet_id": winner["pet_id"],
            "final_pet_name": winner["pet_name"],
        }

    # ==================================================================
    # 6. ASSIGN ONE EVENT
    # ==================================================================
    def assign_event(self, row: pd.Series) -> Dict[str, Any]:
        if not self.pet_profiles:
            return {
                "serial": row.get("serial", ""),
                "user_id": row.get("user_id", ""),
                "visit_date": row.get("visit_date", ""),
                "visit_time": row.get("visit_time", ""),
                "visit_hour": row.get("visit_hour", None),
                "scale_weight": row.get("scale_weight", None),
                "predicted_pet_id": "UNASSIGNED",
                "predicted_pet_name": "No Profiles",
                "final_pet_id": "UNASSIGNED",
                "final_pet_name": "No Profiles",
                "abstained": 1,
                "abstain_reason": "NO_PROFILES",
                "confidence": 0.0,
            }

        weight_lbs = float(row["scale_weight"])
        visit_hour = int(row["visit_hour"])

        ambiguity = self.weight_ambiguity(weight_lbs)
        effective_boost = self.hour_boost * (0.3 + 0.7 * ambiguity)
        baseline = 1.0 / 24.0

        scores: List[Dict[str, Any]] = []
        total_score = 0.0

        for pet_id, profile in self.pet_profiles.items():
            if not profile.get("is_active", True):
                continue

            w_score = self.weight_likelihood(weight_lbs, pet_id)
            h_prior = self.hour_probability(pet_id, visit_hour)

            combined = w_score * (1.0 + effective_boost * (h_prior / baseline - 1.0))
            combined *= self.never_visited_penalty(pet_id)
            combined = max(float(combined), 1e-12)
            total_score += combined

            scores.append({
                "pet_id": pet_id,
                "pet_name": profile["pet_name"],
                "score": combined,
                "prior": h_prior,
                "likelihood": w_score,
                "posterior": 0.0,
                "mean": self.effective_anchor(pet_id),
                "std": self.weight_std(pet_id),
                "zscore": self.weight_zscore(weight_lbs, pet_id),
                "window_size": profile["total_visits"],
                "is_winner": False,
            })

        if total_score > 0:
            for s in scores:
                s["posterior"] = float(s["score"] / total_score)
        elif scores:
            uniform = 1.0 / len(scores)
            for s in scores:
                s["posterior"] = uniform

        scores.sort(key=lambda x: x["posterior"], reverse=True)
        winner = scores[0]
        winner["is_winner"] = True
        second_best = scores[1] if len(scores) > 1 else None

        confidence_metrics = self.calculate_confidence_metrics(scores)
        abstention = self.apply_abstention_logic(
            winner=winner,
            second_best=second_best,
            confidence_metrics=confidence_metrics,
            ambiguity=ambiguity,
            weight_lbs=weight_lbs,
        )

        # Update only if not abstained so uncertain predictions do not contaminate learning
        if self.online_learning and not abstention["abstained"]:
            pet_id = winner["pet_id"]
            self.record_hour(pet_id, visit_hour, weight=1.0)
            self.record_visit_weight(pet_id, weight_lbs)
            self.assigned_visit_counts[pet_id] = self.assigned_visit_counts.get(pet_id, 0) + 1

        return {
            "serial": row.get("serial", ""),
            "user_id": row.get("user_id", ""),
            "visit_date": row.get("visit_date", ""),
            "visit_time": row.get("visit_time", ""),
            "visit_hour": visit_hour,
            "scale_weight": weight_lbs,

            "predicted_pet_id": winner["pet_id"],
            "predicted_pet_name": winner["pet_name"],
            "final_pet_id": abstention["final_pet_id"],
            "final_pet_name": abstention["final_pet_name"],

            "abstained": int(abstention["abstained"]),
            "abstain_reason": abstention["abstain_reason"],

            "confidence": confidence_metrics["confidence"],
            "second_best_pet_id": second_best["pet_id"] if second_best else "",
            "second_best_pet_name": second_best["pet_name"] if second_best else "",
            "second_best_confidence": confidence_metrics["second_best_confidence"],
            "confidence_margin": confidence_metrics["confidence_margin"],
            "score_entropy": confidence_metrics["score_entropy"],
            "normalized_entropy": confidence_metrics["normalized_entropy"],

            "ambiguity": ambiguity,
            "hour_boost_effective": effective_boost,
            "winner_weight_likelihood": winner["likelihood"],
            "winner_hour_prior": winner["prior"],
            "winner_mean": winner["mean"],
            "winner_std": winner["std"],
            "winner_zscore": winner["zscore"],
        }

    # ==================================================================
    # 7. RUN TAKES DATAFRAME AS INPUT
    # ==================================================================
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run algorithm on an already prepared or raw DataFrame.

        Important:
            This function does NOT read CSV.
            Input is DataFrame.
            Output is DataFrame.

        You can call:
            df = algo.datareadingFunction("1000156.csv")
            result_df = algo.run(df)

        Or:
            raw_df = pd.read_csv("1000156.csv")
            df = algo.prepare_dataframe(raw_df)
            result_df = algo.run(df)
        """
        required_prepared_cols = ["visit_datetime", "visit_hour", "timestamp"]
        if not all(c in df.columns for c in required_prepared_cols):
            df = self.prepare_dataframe(df)
        else:
            df = df.copy().sort_values("visit_datetime").reset_index(drop=True)

        self.build_pet_profiles(df)

        output_rows: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            prediction = self.assign_event(row)

            true_pet_id = str(row.get("true_label_pet_id", "")).strip()
            true_cat_name = str(row.get("true_label_cat_name", "")).strip()

            prediction["true_label_pet_id"] = true_pet_id
            prediction["true_label_cat_name"] = true_cat_name

            prediction["is_correct_raw_prediction"] = (
                int(prediction["predicted_pet_id"] == true_pet_id)
                if true_pet_id
                else 0
            )

            # Abstained output is counted as not correct in final accuracy.
            prediction["is_correct_final"] = (
                int(prediction["final_pet_id"] == true_pet_id)
                if true_pet_id
                else 0
            )

            output_rows.append(prediction)

        result_df = pd.DataFrame(output_rows)

        numeric_cols = result_df.select_dtypes(include=["float", "float64", "float32"]).columns
        result_df[numeric_cols] = result_df[numeric_cols].round(6)

        return result_df

    def save_results(self, result_df: pd.DataFrame, output_csv: str) -> None:
        """
        Save result DataFrame as CSV.
        """
        result_df.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")

    # ==================================================================
    # 8. SUMMARY / DIAGNOSTICS
    # ==================================================================
    def summarize_results(self, result_df: pd.DataFrame) -> Dict[str, Any]:
        total = len(result_df)
        abstained = int(result_df["abstained"].sum()) if total and "abstained" in result_df else 0
        non_abstained_df = result_df[result_df["abstained"] == 0] if total else result_df

        raw_accuracy = float(result_df["is_correct_raw_prediction"].mean()) if total else 0.0
        final_accuracy_all = float(result_df["is_correct_final"].mean()) if total else 0.0
        coverage = float(len(non_abstained_df) / total) if total else 0.0
        accuracy_on_covered = (
            float(non_abstained_df["is_correct_final"].mean())
            if len(non_abstained_df) > 0
            else 0.0
        )

        abstain_reason_counts = (
            result_df.loc[result_df["abstained"] == 1, "abstain_reason"]
            .value_counts()
            .to_dict()
            if total and "abstain_reason" in result_df
            else {}
        )

        return {
            "total_events": total,
            "abstained_events": abstained,
            "coverage": coverage,
            "raw_prediction_accuracy_before_abstention": raw_accuracy,
            "final_accuracy_counting_abstain_as_wrong": final_accuracy_all,
            "accuracy_on_non_abstained_events": accuracy_on_covered,
            "abstain_reason_counts": abstain_reason_counts,
        }

    def print_summary(self, result_df: pd.DataFrame) -> None:
        summary = self.summarize_results(result_df)

        print("\n================ Circadian Weight Result Summary ================")
        print(f"Total events                              : {summary['total_events']}")
        print(f"Abstained events                          : {summary['abstained_events']}")
        print(f"Coverage                                  : {summary['coverage']:.4f}")
        print(f"Raw prediction accuracy before abstention : {summary['raw_prediction_accuracy_before_abstention']:.4f}")
        print(f"Final accuracy, abstain counted wrong     : {summary['final_accuracy_counting_abstain_as_wrong']:.4f}")
        print(f"Accuracy on non-abstained events          : {summary['accuracy_on_non_abstained_events']:.4f}")

        print("\nAbstain reason counts:")
        if summary["abstain_reason_counts"]:
            for reason, count in summary["abstain_reason_counts"].items():
                print(f"  {reason}: {count}")
        else:
            print("  None")
        print("=================================================================\n")

    def get_circadian_profiles(self) -> pd.DataFrame:
        rows = []
        for pet_id, p in self.pet_profiles.items():
            hour_counts = p["hour_counts"]
            peak_hours = sorted(range(24), key=lambda h: hour_counts[h], reverse=True)[:3]
            rows.append({
                "pet_id": pet_id,
                "pet_name": p["pet_name"],
                "weight_anchor": round(float(p["weight_anchor"]), 6),
                "ema_weight": round(float(p["ema_weight"]), 6),
                "weight_std": round(float(self.weight_std(pet_id)), 6),
                "total_circadian_visits": p["total_visits"],
                "online_weight_visits": p["visit_count"],
                "peak_hours": str(peak_hours),
                "hour_counts": str([round(float(x), 2) for x in hour_counts]),
            })
        return pd.DataFrame(rows)

    def save_circadian_profiles(self, output_csv: str) -> None:
        profiles_df = self.get_circadian_profiles()
        profiles_df.to_csv(output_csv, index=False)
        print(f"Circadian profiles saved to: {output_csv}")


# ======================================================================
# USAGE EXAMPLE
# ======================================================================
if __name__ == "__main__":
    # ---------------------------------------------------------------
    # Example 1: Read CSV using datareadingFunction(), then run(df)
    # ---------------------------------------------------------------
    input_csv_path = "1000156.csv"
    output_csv_path = "1000156_circadian_confidence_abstention_results.csv"
    profile_output_csv_path = "1000156_circadian_profiles.csv"

    algo = CircadianWeightAlgorithm(
        sigma=0.4,
        hour_boost=0.5,
        close_weight_threshold=1.0,
        reassignment_weight=3.0,
        ema_alpha=0.15,
        warm_start_with_true_labels=True,
        online_learning=True,

        # Confidence + abstention thresholds
        enable_abstention=True,
        min_confidence=0.60,
        min_confidence_margin=0.15,
        high_ambiguity_threshold=0.50,
        max_weight_zscore=3.0,
        min_winner_likelihood=0.05,
    )

    # Step 1: Read CSV and get prepared DataFrame
    df = algo.datareadingFunction(input_csv_path)

    # Step 2: Run algorithm on DataFrame, not CSV path
    result_df = algo.run(df)

    # Step 3: Store output DataFrame as CSV
    algo.save_results(result_df, output_csv_path)

    # Step 4: Save learned circadian profiles also, optional
    algo.save_circadian_profiles(profile_output_csv_path)

    # Step 5: Print summary
    algo.print_summary(result_df)

    # ---------------------------------------------------------------
    # Example 2: If you already loaded df yourself
    # ---------------------------------------------------------------
    # raw_df = pd.read_csv(input_csv_path)
    # prepared_df = algo.prepare_dataframe(raw_df)
    # result_df = algo.run(prepared_df)
    # result_df.to_csv(output_csv_path, index=False)
