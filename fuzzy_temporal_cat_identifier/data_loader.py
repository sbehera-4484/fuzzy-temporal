import pandas as pd
from typing import Dict

from config import FuzzyTemporalConfig


class CatVisitDataLoader:
    """
    Handles CSV loading and DataFrame preparation.

    Design rule:
    - File paths are accepted only by this class.
    - Diagnostic and prediction classes receive prepared DataFrames only.
    """

    REQUIRED_COLUMNS = [
        "user_id",
        "visit_date",
        "visit_time",
        "true_label_cat_name",
        "profile_weight",
        "scale_weight",
    ]

    def __init__(self, config: FuzzyTemporalConfig):
        self.config = config

    def load_csv(self, input_file: str) -> pd.DataFrame:
        """Load input CSV into a raw DataFrame."""
        return pd.read_csv(input_file)

    def validate_columns(self, df: pd.DataFrame) -> None:
        """Validate required input columns before preprocessing."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add event_timestamp and event_date.

        Expected columns:
        - visit_date
        - visit_time
        """
        self.validate_columns(df)

        df = df.copy()
        df["event_timestamp"] = pd.to_datetime(
            df["visit_date"].astype(str) + " " + df["visit_time"].astype(str),
            errors="coerce",
        )
        df["event_date"] = df["event_timestamp"].dt.date
        return df

    def add_enrollment_id(self, df_user: pd.DataFrame) -> pd.DataFrame:
        """
        Create label-free enrollment id from profile_weight.

        IMPORTANT:
        true_label_cat_name is not used here.
        """
        df_user = df_user.copy()
        round_dp = self.config.enroll_round_dp

        df_user["enroll_id"] = (
            "cat_"
            + df_user["profile_weight"]
            .round(round_dp)
            .map(("{: ." + str(round_dp) + "f}kg").format)
            .str.replace(" ", "", regex=False)
        )

        return df_user

    def get_user_dataframe(self, df: pd.DataFrame, user_id) -> pd.DataFrame:
        """Return cleaned prepared rows for one user."""
        df_user = df[df["user_id"] == user_id].copy()
        df_user = df_user.sort_values("event_timestamp")

        if df_user.empty:
            return df_user

        df_user = df_user.dropna(
            subset=[
                "event_timestamp",
                "true_label_cat_name",
                "profile_weight",
                "scale_weight",
                "event_date",
            ]
        )
        return df_user

    def build_enrollment_catalog(self, df_user: pd.DataFrame) -> Dict[str, float]:
        """Return enroll_id -> mean profile weight in kg."""
        return df_user.groupby("enroll_id")["profile_weight"].mean().to_dict()

    def map_enrollment_to_names(self, df_user: pd.DataFrame) -> Dict[str, str]:
        """
        KPI-only mapping.

        Returns:
            enroll_id -> most frequent true_label_cat_name
        """
        return (
            df_user.groupby("enroll_id")["true_label_cat_name"]
            .agg(lambda s: s.value_counts().idxmax())
            .to_dict()
        )
