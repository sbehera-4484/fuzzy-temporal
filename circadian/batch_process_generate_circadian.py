import os
from pathlib import Path
import pandas as pd

from circadian_weight_algorithm import CircadianWeightAlgorithm


def get_user_folders(base_dir):
    user_folders = []

    for user_folder in Path(base_dir).iterdir():

        if user_folder.is_dir():

            user_id = user_folder.name

            csv_file = user_folder / f"{user_id}.csv"

            if csv_file.exists():

                user_folders.append(
                    (user_id, str(csv_file))
                )

    return sorted(user_folders)


def process_user(user_id, csv_file):

    try:

        algo = CircadianWeightAlgorithm(
            sigma=0.4,
            hour_boost=0.5,
            close_weight_threshold=1.0,
            reassignment_weight=3.0,
            ema_alpha=0.15,
            warm_start_with_true_labels=False,
            online_learning=True,
            enable_abstention=True,
            min_confidence=0.30,
            min_confidence_margin=0.15,
            high_ambiguity_threshold=0.50,
            max_weight_zscore=3.0,
            min_winner_likelihood=0.05,
        )

        df = algo.datareadingFunction(csv_file)

        results = algo.run(df)

        summary = algo.summarize_results(results)

        return {
            "user_id": user_id,
            "total_events":
                summary.get("total_events", 0),

            "abstained_events":
                summary.get("abstained_events", 0),

            "coverage":
                summary.get("coverage", 0),

            "accuracy":
                summary.get(
                    "final_accuracy_counting_abstain_as_wrong",
                    0
                ),

            "selective_accuracy":
                summary.get(
                    "final_accuracy_excluding_abstains",
                    None
                ),
        }

    except Exception as e:

        print(
            f"ERROR processing {user_id}: {e}"
        )

        return {
            "user_id": user_id,
            "total_events": 0,
            "abstained_events": 0,
            "coverage": 0,
            "accuracy": 0,
            "selective_accuracy": None,
            "error": str(e),
        }


def batch_process(base_dir, output_csv):

    user_folders = get_user_folders(base_dir)

    print(
        f"Found {len(user_folders)} users"
    )

    all_results = []

    for idx, (user_id, csv_file) in enumerate(
        user_folders,
        start=1
    ):

        print(
            f"[{idx}/{len(user_folders)}] "
            f"Processing {user_id}"
        )

        user_summary = process_user(
            user_id,
            csv_file
        )

        all_results.append(
            user_summary
        )

    results_df = pd.DataFrame(
        all_results
    )

    results_df = results_df.sort_values(
        "accuracy",
        ascending=True
    )

    results_df.to_csv(
        output_csv,
        index=False
    )

    print("\nSaved:")
    print(output_csv)

    print("\nOverall statistics")
    print(
        f"Users: {len(results_df)}"
    )

    print(
        f"Mean accuracy: "
        f"{results_df['accuracy'].mean():.4f}"
    )

    print(
        f"Median accuracy: "
        f"{results_df['accuracy'].median():.4f}"
    )

    print(
        f"Min accuracy: "
        f"{results_df['accuracy'].min():.4f}"
    )

    print(
        f"Max accuracy: "
        f"{results_df['accuracy'].max():.4f}"
    )

    print(
        f"Accuracy < 0.85 : "
        f"{(results_df['accuracy'] < 0.85).sum()}"
    )


if __name__ == "__main__":

    base_directory = (
        r"C:\Users\shreetam_behera\Downloads"
        r"\WBID_Scenarios"
        r"\WBID_MultipleCats_Overlapping_Dataset"
    )

    output_csv = (
        r"C:\Users\shreetam_behera\Downloads"
        r"\WBID_Scenarios"
        r"\WBID_MultipleCats_Overlapping_Dataset"
        r"\all_circadian_users_accuracy_summary.csv"
    )

    batch_process(
        base_directory,
        output_csv
    )