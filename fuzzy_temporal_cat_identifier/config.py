from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class FuzzyTemporalConfig:
    """
    Configuration for fuzzy temporal cat identification.

    Notes
    -----
    - base_sigma_lb, min_dynamic_sigma_lb and max_dynamic_sigma_lb are in pounds.
    - All model computations internally use kg.
    - true_label_cat_name must be used only for KPI/post-processing, not prediction.
    """

    # Output
    root_output_folder: str | None = None

    # Unit conversion
    lb_to_kg: float = 0.45359237

    # Sigma configuration. These are intentionally stored in pounds.
    base_sigma_lb: float = 0.15
    min_dynamic_sigma_lb: float = 0.15
    max_dynamic_sigma_lb: float = 5.0
    sigma_iterations: int = 8

    # Temporal behavior
    temporal_weight: float = 0.40
    state_alpha: float = 0.60
    decay_minutes: float = 180.0

    # Data columns
    device_col: str = "device_serial"
    enroll_round_dp: int = 2

    # Prediction control flags
    use_dynamic_sigma_in_prediction: bool = True
    use_overlap_penalty_in_prediction: bool = True
    use_consistency_penalty_in_prediction: bool = True

    # Penalty configuration
    overlap_penalty_strength: float = 0.35
    consistency_penalty_strength: float = 0.30
    min_score_factor: float = 0.05

    # Diagnostic thresholds
    high_overlap_score: float = 0.70
    medium_overlap_score: float = 0.40
    low_consistency_score: float = 0.50
    medium_consistency_score: float = 0.70

    @property
    def kg_to_lb(self) -> float:
        return 1.0 / self.lb_to_kg

    @property
    def base_sigma_kg(self) -> float:
        return self.base_sigma_lb * self.lb_to_kg

    @property
    def min_dynamic_sigma_kg(self) -> float:
        return self.min_dynamic_sigma_lb * self.lb_to_kg

    @property
    def max_dynamic_sigma_kg(self) -> float:
        return self.max_dynamic_sigma_lb * self.lb_to_kg

    def to_dataframe(self) -> pd.DataFrame:
        """Return configuration as a simple parameter/value dataframe."""
        return pd.DataFrame(
            [{"parameter": key, "value": value} for key, value in asdict(self).items()]
        )
