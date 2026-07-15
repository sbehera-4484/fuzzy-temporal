from datetime import datetime
import pprint


def generate_mongo_doc(
    visit_id,
    user_id,
    serial,
    weight,
    duration,
    scenario,
    algorithm_outputs
):
    """
    algorithm_outputs format:

    {
        "gmm_k": {
            "pet_id": "...",
            "confidence": 0.53,
            "status": "ASSIGNED"
        },
        ...
    }
    """

    # ------------------------------
    # Find winning algorithm
    # ------------------------------
    best_algo = max(
        algorithm_outputs.items(),
        key=lambda x: x[1]["confidence"]
    )

    best_algo_name = best_algo[0]
    best_result = best_algo[1]

    assigned_pet_id = best_result["pet_id"]
    ensemble_confidence = best_result["confidence"]

    # ------------------------------
    # Build confidence section
    # ------------------------------
    confidence_dict = {}

    for algo_name, result in algorithm_outputs.items():
        confidence_dict[algo_name] = result["confidence"]

    confidence_dict["ensemble"] = ensemble_confidence

    # ------------------------------
    # Generate document
    # ------------------------------
    mongo_doc = {
        "_id": "GENERATED_OBJECT_ID",
        "visit_id": visit_id,
        "algo_name": "ensemble",

        "algo_outputs": algorithm_outputs,

        "assigned_pet_id": assigned_pet_id,

        "confidence": confidence_dict,

        "created_ts": datetime.utcnow().isoformat(),
        "scenario": scenario,
        "serial": serial,
        "status": best_result["status"],
        "updated_ts": datetime.utcnow().isoformat(),

        "usage_report": {
            "weight": weight,
            "duration": duration
        },

        "user_id": user_id,
        "visit_ts": datetime.utcnow().isoformat(),
        "weight": weight
    }

    return mongo_doc


# ==========================================================
# INPUT DATA
# ==========================================================

algorithm_outputs = {
    "gmm_k": {
        "pet_id": "PET-CAT-001",
        "confidence": 0.5336,
        "status": "ASSIGNED"
    },

    "gmm_fuzzy_linkage": {
        "pet_id": "PET-CAT-002",
        "confidence": 0.5883,
        "status": "ASSIGNED"
    },

    "gaussian_nb": {
        "pet_id": "PET-CAT-002",
        "confidence": 0.6721,
        "status": "ASSIGNED"
    },

    "temporal_fuzzy": {
        "pet_id": "PET-CAT-002",
        "confidence": 0.7412,
        "status": "ASSIGNED"
    }
}

# ==========================================================
# GENERATE DOCUMENT
# ==========================================================

mongo_doc = generate_mongo_doc(
    visit_id="ae9f2aff-39db-4a30-a293-0f5ef62197qw",
    user_id="jtipton",
    serial="LR5-01-01-01-2508-270006",
    weight=12.5,
    duration=134,
    scenario="HAS_HISTORY",
    algorithm_outputs=algorithm_outputs
)

# ==========================================================
# OUTPUT
# ==========================================================

print("\nGenerated MongoDB Document:\n")
pprint.pp(mongo_doc)

print("\nAssigned Pet:", mongo_doc["assigned_pet_id"])
print("Winning Confidence:", mongo_doc["confidence"]["ensemble"])

# Modification for pet_id voting
from collections import Counter


def generate_ensemble(mongo_doc):
    """
    Majority Vote Ensemble

    Selection Logic:
    1. Select pet_id with highest frequency.
    2. Tie-break using sum of confidences.
    3. For winning pet_id, select algorithm with highest confidence.

    Output:
    {
        "assigned_pet_id": ...,
        "selected_algorithm": ...,
        "confidence": ...,
        "vote_count": ...,
        "total_algorithms": ...,
        "weighted_score": ...,
        "status": ...
    }
    """

    algo_outputs = mongo_doc["algo_outputs"]

    total_algorithms = len(algo_outputs)

    # Vote count per pet
    pet_votes = Counter()

    # Sum confidence per pet
    pet_scores = {}

    for algo_name, result in algo_outputs.items():
        pet_id = result["pet_id"]
        confidence = result["confidence"]

        pet_votes[pet_id] += 1
        pet_scores[pet_id] = pet_scores.get(pet_id, 0.0) + confidence

    # -------------------------------------------------
    # Step 1: Find highest vote count
    # -------------------------------------------------
    max_votes = max(pet_votes.values())

    candidate_pets = [
        pet_id
        for pet_id, votes in pet_votes.items()
        if votes == max_votes
    ]

    # -------------------------------------------------
    # Step 2: Tie-break using weighted confidence score
    # -------------------------------------------------
    assigned_pet_id = max(
        candidate_pets,
        key=lambda pet: pet_scores[pet]
    )

    vote_count = pet_votes[assigned_pet_id]
    weighted_score = pet_scores[assigned_pet_id]

    # -------------------------------------------------
    # Step 3: Among algorithms voting
    #         for winning pet, select highest confidence
    # -------------------------------------------------
    selected_algorithm = None
    best_confidence = -1

    for algo_name, result in algo_outputs.items():

        if result["pet_id"] == assigned_pet_id:

            if result["confidence"] > best_confidence:
                best_confidence = result["confidence"]
                selected_algorithm = algo_name

    ensemble = {
        "assigned_pet_id": assigned_pet_id,
        "selected_algorithm": selected_algorithm,
        "confidence": round(best_confidence, 4),
        "vote_count": vote_count,
        "total_algorithms": total_algorithms,
        "weighted_score": round(weighted_score, 4),
        "status": "ASSIGNED"
    }

    return ensemble


# Usage
ensemble = generate_ensemble(mongo_doc)
print(ensemble)