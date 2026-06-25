from typing import Dict
import pandas as pd
from config import FuzzyTemporalTriangularConfig


class CatVisitDataLoader:
    """Handles CSV loading and DataFrame preparation."""
    REQUIRED_COLUMNS = ["user_id", "visit_date", "visit_time", "true_label_cat_name", "profile_weight", "scale_weight"]

    def __init__(self, config: FuzzyTemporalTriangularConfig):
        self.config = config

    def load_csv(self, input_file: str) -> pd.DataFrame:
        return pd.read_csv(input_file)

    def validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_columns(df)
        df = df.copy()
        df["event_timestamp"] = pd.to_datetime(
            df["visit_date"].astype(str) + " " + df["visit_time"].astype(str), errors="coerce"
        )
        df["event_date"] = df["event_timestamp"].dt.date
        return df

    def get_user_dataframe(self, df: pd.DataFrame, user_id) -> pd.DataFrame:
        df_user = df[df["user_id"] == user_id].copy().sort_values("event_timestamp")
        if df_user.empty:
            return df_user
        return df_user.dropna(subset=["event_timestamp", "true_label_cat_name", "profile_weight", "scale_weight", "event_date"])

    def add_enrollment_id(self, df_user: pd.DataFrame) -> pd.DataFrame:
        df_user = df_user.copy()
        dp = self.config.enroll_round_dp
        df_user["enroll_id"] = (
            "cat_" + df_user["profile_weight"].round(dp).map(("{: ." + str(dp) + "f}kg").format).str.replace(" ", "", regex=False)
        )
        return df_user

    def build_enrollment_catalog(self, df_user: pd.DataFrame) -> Dict[str, float]:
        return df_user.groupby("enroll_id")["profile_weight"].mean().to_dict()

    def map_enrollment_to_names(self, df_user: pd.DataFrame) -> Dict[str, str]:
        return df_user.groupby("enroll_id")["true_label_cat_name"].agg(lambda s: s.value_counts().idxmax()).to_dict()
