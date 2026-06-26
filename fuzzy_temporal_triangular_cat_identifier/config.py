from dataclasses import dataclass, asdict, field
import pandas as pd


@dataclass
class FuzzyTemporalTriangularConfig:
    """Configuration for fuzzy temporal cat identification with triangular sigma-band selection."""
    root_output_folder: str | None = None
    lb_to_kg: float = 0.45359237
    base_sigma_lb: float = 0.15
    min_dynamic_sigma_lb: float = 0.15
    max_dynamic_sigma_lb: float = 5.0
    sigma_iterations: int = 8
    temporal_weight: float = 0.40
    state_alpha: float = 0.60
    decay_minutes: float = 180.0
    device_col: str = "device_serial"
    enroll_round_dp: int = 2
    use_dynamic_sigma_in_prediction: bool = True
    use_overlap_penalty_in_prediction: bool = True
    use_consistency_penalty_in_prediction: bool = True
    overlap_penalty_strength: float = 0.35
    consistency_penalty_strength: float = 0.30
    min_score_factor: float = 0.05
    high_overlap_score: float = 0.70
    medium_overlap_score: float = 0.40
    low_consistency_score: float = 0.50
    medium_consistency_score: float = 0.70
    
    # === ABSTENTION LOGIC ADDITIONS ===
    use_abstention: bool = True
    abstain_label: str = "unknown"
    confidence_threshold: float = 0.50
    overlap_abstention_threshold: float = 0.70
    score_gap_threshold: float = 0.10
    # ==================================

    sigma_bands_lb: dict = field(default_factory=lambda: {
        "sigma_0_5_to_1_0_lb": {"left": 0.5, "center": 0.75, "right": 1.0, "representative": 0.75},
        "sigma_1_0_to_2_0_lb": {"left": 1.0, "center": 1.5, "right": 2.0, "representative": 1.5},
        "sigma_2_0_to_5_0_lb": {"left": 2.0, "center": 3.5, "right": 5.0, "representative": 3.5},
        "sigma_ge_5_0_lb": {"left": 5.0, "center": 6.0, "right": None, "representative": 5.0},
    })

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
        records=[]
        for key, value in asdict(self).items():
            if key == "sigma_bands_lb":
                for band_name, params in value.items():
                    for p_key, p_val in params.items():
                        records.append({"parameter": f"sigma_bands_lb.{band_name}.{p_key}", "value": p_val})
            else:
                records.append({"parameter": key, "value": value})
        return pd.DataFrame(records)