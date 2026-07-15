"""
confidence_weighted_ewma_profile_update.py

Purpose
-------
Update cat profile weights from 24-hour temporal weight observations using
Confidence-Weighted EWMA.

Expected event dataframe columns:
    cat_id      : unique cat identifier
    weight      : observed event/stable weight
    confidence  : confidence/probability that this event belongs to cat_id

Expected profile dataframe columns:
    cat_id
    profile_weight
    profile_std        optional; if missing, default_std is used

Output:
    Updated profile dataframe with profile_weight, profile_std, samples_used,
    daily_weight_estimate, daily_std_estimate.

This method supports both profile weight increase and decrease:
    new_profile = old_profile + effective_alpha * (daily_estimate - old_profile)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]

    if len(values) == 0:
        return np.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative_weight = np.cumsum(weights)
    cutoff = 0.5 * np.sum(weights)
    return float(values[np.searchsorted(cumulative_weight, cutoff)])


def weighted_mad(values: np.ndarray, weights: np.ndarray, center: float) -> float:
    """
    Confidence-weighted robust std estimate using MAD.
    Gaussian-consistent scale factor = 1.4826.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]

    if len(values) == 0 or not np.isfinite(center):
        return np.nan

    deviations = np.abs(values - center)
    mad = weighted_median(deviations, weights)
    return float(1.4826 * mad)


def adaptive_alpha(
    old_weight: float,
    daily_weight: float,
    base_alpha: float = 0.05,
    avg_confidence: float = 1.0,
    small_delta: float = 0.05,
    medium_delta: float = 0.20,
    min_alpha_factor: float = 0.25,
) -> float:
    """
    Reduce update speed for large jumps, because large jumps are often caused by
    misclassification, partial scale contact, posture, or sensor noise.

    Increase and decrease are both supported because delta can be positive or negative.
    """
    delta = abs(float(daily_weight) - float(old_weight))

    if delta <= small_delta:
        stability_factor = 1.0
    elif delta <= medium_delta:
        stability_factor = 0.5
    else:
        stability_factor = min_alpha_factor

    avg_confidence = float(np.clip(avg_confidence, 0.0, 1.0))
    return float(base_alpha * stability_factor * avg_confidence)


def update_profiles_confidence_weighted_ewma(
    events_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    cat_col: str = "cat_id",
    weight_col: str = "weight",
    confidence_col: str = "confidence",
    profile_weight_col: str = "profile_weight",
    profile_std_col: str = "profile_std",
    base_alpha: float = 0.05,
    std_alpha: float | None = None,
    min_confidence: float = 0.70,
    min_samples: int = 3,
    default_std: float = 0.10,
    use_weighted_median: bool = True,
) -> pd.DataFrame:
    """
    Update one profile row per cat using 24-hour observations.

    Formula:
        daily_estimate = confidence-weighted median/mean of today's observations
        effective_alpha = base_alpha * confidence_factor * stability_factor
        new_weight = old_weight + effective_alpha * (daily_estimate - old_weight)

    This naturally moves profile weight upward or downward depending on delta.
    """
    required_event_cols = {cat_col, weight_col, confidence_col}
    missing_events = required_event_cols - set(events_df.columns)
    if missing_events:
        raise ValueError(f"events_df missing columns: {sorted(missing_events)}")

    required_profile_cols = {cat_col, profile_weight_col}
    missing_profiles = required_profile_cols - set(profiles_df.columns)
    if missing_profiles:
        raise ValueError(f"profiles_df missing columns: {sorted(missing_profiles)}")

    if std_alpha is None:
        std_alpha = base_alpha

    events = events_df.copy()
    profiles = profiles_df.copy()

    if profile_std_col not in profiles.columns:
        profiles[profile_std_col] = default_std

    events = events[
        events[weight_col].notna()
        & events[confidence_col].notna()
        & (events[confidence_col] >= min_confidence)
    ].copy()

    updated_rows = []

    for _, profile_row in profiles.iterrows():
        cat_id = profile_row[cat_col]
        old_weight = float(profile_row[profile_weight_col])
        old_std = float(profile_row.get(profile_std_col, default_std))

        cat_events = events[events[cat_col] == cat_id]
        samples_used = len(cat_events)

        output_row = profile_row.to_dict()
        output_row["samples_used"] = samples_used
        output_row["daily_weight_estimate"] = np.nan
        output_row["daily_std_estimate"] = np.nan
        output_row["effective_alpha"] = 0.0
        output_row["update_status"] = "not_updated"

        if samples_used < min_samples:
            updated_rows.append(output_row)
            continue

        weights = cat_events[weight_col].astype(float).to_numpy()
        confs = cat_events[confidence_col].astype(float).to_numpy()

        if use_weighted_median:
            daily_weight = weighted_median(weights, confs)
        else:
            daily_weight = float(np.average(weights, weights=confs))

        daily_std = weighted_mad(weights, confs, daily_weight)
        if not np.isfinite(daily_std) or daily_std <= 0:
            daily_std = old_std

        avg_confidence = float(np.mean(confs))
        eff_alpha = adaptive_alpha(
            old_weight=old_weight,
            daily_weight=daily_weight,
            base_alpha=base_alpha,
            avg_confidence=avg_confidence,
        )

        new_weight = old_weight + eff_alpha * (daily_weight - old_weight)
        new_std = old_std + std_alpha * (daily_std - old_std)

        output_row[profile_weight_col] = float(new_weight)
        output_row[profile_std_col] = float(max(new_std, 1e-6))
        output_row["daily_weight_estimate"] = float(daily_weight)
        output_row["daily_std_estimate"] = float(daily_std)
        output_row["effective_alpha"] = float(eff_alpha)
        output_row["avg_confidence"] = avg_confidence
        output_row["update_status"] = "updated"

        updated_rows.append(output_row)

    return pd.DataFrame(updated_rows)


if __name__ == "__main__":
    # Example 24-hour event data
    events_df = pd.DataFrame({
        "cat_id": ["cat_A", "cat_A", "cat_A", "cat_A", "cat_B", "cat_B", "cat_B"],
        "weight": [5.10, 5.08, 5.12, 4.85, 6.20, 6.18, 6.24],
        "confidence": [0.95, 0.90, 0.92, 0.55, 0.96, 0.91, 0.93],
    })

    profiles_df = pd.DataFrame({
        "cat_id": ["cat_A", "cat_B"],
        "profile_weight": [5.00, 6.10],
        "profile_std": [0.10, 0.12],
    })

    updated_profiles = update_profiles_confidence_weighted_ewma(
        events_df=events_df,
        profiles_df=profiles_df,
        base_alpha=0.05,
        min_confidence=0.70,
        min_samples=2,
        use_weighted_median=True,
    )

    print(updated_profiles)
    # updated_profiles.to_csv("updated_profiles_ewma.csv", index=False)
