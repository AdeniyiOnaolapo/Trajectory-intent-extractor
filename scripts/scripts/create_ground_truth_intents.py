import csv
from pathlib import Path

from selection.traj_intent import extract_traj_intent
from your_project_loader import load_candidate_clips  # replace with your actual loader


output = Path("outputs/ground_truth_intents.csv")
output.parent.mkdir(parents=True, exist_ok=True)

columns = [
    "clip_id",
    "longitudinal_intent",
    "lateral_intent",
    "minimum_speed_mps",
    "final_speed_mps",
    "speed_change_mps",
    "heading_change_deg",
    "lateral_offset_m",
]

with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

    for clip_id, trajectory, current_speed in load_candidate_clips():
        result = extract_traj_intent(trajectory, current_speed)
        writer.writerow({
            "clip_id": clip_id,
            "longitudinal_intent": result["longitudinal_intent"],
            "lateral_intent": result["lateral_intent"],
            **result["diagnostics"],
        })

print(f"Wrote {output}")
