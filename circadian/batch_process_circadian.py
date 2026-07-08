"""
Batch process CircadianWeightAlgorithm for all user IDs in WBID_MultipleCats_Overlapping_Dataset.

This script:
1. Iterates through all subfolders in the dataset directory
2. For each subfolder (user_id), finds the corresponding CSV file
3. Runs CircadianWeightAlgorithm on the CSV
4. Saves results to the same subfolder
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
import pandas as pd

from circadian_weight_algorithm import CircadianWeightAlgorithm


def get_user_folders(base_dir: str) -> List[Tuple[str, str]]:
    """
    Get all user ID folders and their corresponding CSV files.
    
    Returns:
        List of tuples: (user_id, csv_file_path)
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        return []
    
    user_folders = []
    
    # Iterate through all subdirectories
    for user_folder in base_path.iterdir():
        if user_folder.is_dir():
            user_id = user_folder.name
            csv_file = user_folder / f"{user_id}.csv"
            print(csv_file)
            
            if csv_file.exists():
                user_folders.append((user_id, str(csv_file)))
            else:
                print(f"Warning: CSV file not found for user {user_id}: {csv_file}")
    
    return sorted(user_folders)


def process_single_user(
    user_id: str,
    csv_file_path: str,
    output_dir: str,
    algo_params: dict = None
) -> Tuple[bool, str]:
    """
    Process a single user's CSV file with CircadianWeightAlgorithm.
    
    Returns:
        Tuple: (success: bool, message: str)
    """
    try:
        # Default parameters
        if algo_params is None:
            algo_params = {
                "sigma": 0.4,
                "hour_boost": 0.5,
                "close_weight_threshold": 1.0,
                "reassignment_weight": 3.0,
                "ema_alpha": 0.15,
                "warm_start_with_true_labels": True,
                "online_learning": True,
                "enable_abstention": True,
                "min_confidence": 0.30,
                "min_confidence_margin": 0.15,
                "high_ambiguity_threshold": 0.50,
                "max_weight_zscore": 3.0,
                "min_winner_likelihood": 0.05,
            }
        
        # Create algorithm instance
        algo = CircadianWeightAlgorithm(**algo_params)
        
        # Set up output paths
        output_results_csv = os.path.join(output_dir, f"{user_id}_circadian_confidence_abstention_results.csv")
        output_profiles_csv = os.path.join(output_dir, f"{user_id}_circadian_profiles.csv")
        
        # Run algorithm
        df = algo.datareadingFunction(csv_file_path)
        results = algo.run(df)
        algo.save_results(results, output_results_csv)
        
        # Save profiles
        profiles = algo.get_circadian_profiles()
        profiles.to_csv(output_profiles_csv, index=False)
        
        # Get summary
        summary = algo.summarize_results(results)
        
        message = (
            f"User {user_id}: "
            f"Total={summary['total_events']}, "
            f"Abstained={summary['abstained_events']}, "
            f"Coverage={summary['coverage']:.4f}, "
            f"Accuracy={summary['final_accuracy_counting_abstain_as_wrong']:.4f}"
        )
        
        return True, message
        
    except Exception as e:
        return False, f"User {user_id}: ERROR - {str(e)}"


def batch_process(
    base_dir: str,
    algo_params: dict = None,
    max_users: int = None
) -> None:
    """
    Batch process all users in the dataset directory.
    
    Args:
        base_dir: Path to WBID_MultipleCats_Overlapping_Dataset folder
        algo_params: Dictionary of algorithm parameters (optional)
        max_users: Maximum number of users to process (None = all)
    """
    print("="*80)
    print(f"Batch Processing CircadianWeightAlgorithm")
    print(f"Base Directory: {base_dir}")
    print("="*80)
    
    # Get all user folders
    user_folders = get_user_folders(base_dir)
    
    if not user_folders:
        print("No user folders found!")
        return
    
    if max_users:
        user_folders = user_folders[:max_users]
    
    print(f"\nFound {len(user_folders)} users to process\n")
    
    # Process each user
    successful = 0
    failed = 0
    
    for idx, (user_id, csv_file_path) in enumerate(user_folders, 1):
        output_dir = os.path.dirname(csv_file_path)
        
        success, message = process_single_user(user_id, csv_file_path, output_dir, algo_params)
        
        if success:
            successful += 1
            print(f"[{idx}/{len(user_folders)}] ✓ {message}")
        else:
            failed += 1
            print(f"[{idx}/{len(user_folders)}] ✗ {message}")
    
    # Summary
    print("\n" + "="*80)
    print(f"SUMMARY: {successful} successful, {failed} failed out of {len(user_folders)} users")
    print("="*80)


if __name__ == "__main__":
    # Base directory containing all user folders
    base_directory = r"C:\Users\shreetam_behera\Downloads\WBID_Scenarios\WBID_MultipleCats_Overlapping_Dataset"
    
    # Optional: limit number of users for testing
    # max_users_to_process = 5
    max_users_to_process = None  # None means process all
    
    # Optional: customize algorithm parameters
    # algo_params = {
    #     "sigma": 0.4,
    #     "hour_boost": 0.5,
    #     "min_confidence": 0.30,
    #     # ... add other parameters as needed
    # }
    algo_params = None  # None means use default parameters
    
    # Run batch processing
    batch_process(base_directory, algo_params, max_users_to_process)
