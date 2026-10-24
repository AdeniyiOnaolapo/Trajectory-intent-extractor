# Trajectory-intent-extractor
#Implement selection/traj_intent.py. Input: a 64-waypoint trajectory (10 Hz, ego frame) plus the current ego speed. 
from __future__ import annotations

from typing import Any
import numpy as np


def extract_traj_intent(
    trajectory: np.ndarray,
    current_speed: float,
) -> dict[str, Any]:
    """Classify a 64-waypoint ego-frame trajectory.

    Args:
        trajectory: Array shaped (64, 2) [x, y] or (64, 3) [x, y, z],
            in metres, sampled at 10 Hz.
        current_speed: Current ego speed in m/s.

    Returns:
        Dict with longitudinal and lateral intent labels plus diagnostics.
    """
    traj = np.asarray(trajectory, dtype=float)

    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError("trajectory must have shape (N, 2+) with x and y")
    if traj.shape[0] != 64:
        raise ValueError(f"expected 64 waypoints, got {traj.shape[0]}")
    if not np.isfinite(traj[:, :2]).all() or not np.isfinite(current_speed):
        raise ValueError("trajectory and current_speed must be finite")
    if current_speed < 0:
        raise ValueError("current_speed must be non-negative")

    xy = traj[:, :2]
    # Include the current ego position (origin) when estimating future speeds.
    points = np.vstack((np.zeros((1, 2)), xy))
    step_speeds = np.linalg.norm(np.diff(points, axis=0), axis=1) * 10.0

    min_speed = float(step_speeds.min())
    final_speed = float(step_speeds[-1])
    speed_change = final_speed - float(current_speed)
    decelerates = speed_change < -1.0

    if min_speed < 0.5 and decelerates:
        longitudinal = "STOP"
    elif speed_change < -1.0:
        longitudinal = "DECELERATE"
    elif speed_change > 1.0:
        longitudinal = "ACCELERATE"
    else:
        longitudinal = "MAINTAIN"

    # Signed net heading change, using robust multi-step displacement vectors.
    # atan2 convention: positive is left in the usual x-forward, y-left ego frame.
    first_vec = xy[4] - xy[0]
    last_vec = xy[-1] - xy[-5]

    def heading(vec: np.ndarray) -> float | None:
        if np.linalg.norm(vec) < 1e-6:
            return None
        return float(np.arctan2(vec[1], vec[0]))

    h0, h1 = heading(first_vec), heading(last_vec)
    if h0 is None or h1 is None:
        heading_change_deg = 0.0
    else:
        delta = (h1 - h0 + np.pi) % (2 * np.pi) - np.pi
        heading_change_deg = float(np.degrees(delta))

    lateral_offset = float(xy[-1, 1])

    if heading_change_deg > 30.0:
        lateral = "TURN_LEFT"
    elif heading_change_deg < -30.0:
        lateral = "TURN_RIGHT"
    elif 2.0 <= abs(lateral_offset) <= 5.0 and abs(heading_change_deg) < 20.0:
        lateral = "LANE_CHANGE"
    elif abs(lateral_offset) < 1.0:
        lateral = "KEEP"
    else:
        lateral = "UNKNOWN"

    return {
        "longitudinal_intent": longitudinal,
        "lateral_intent": lateral,
        "diagnostics": {
            "minimum_speed_mps": min_speed,
            "final_speed_mps": final_speed,
            "speed_change_mps": speed_change,
            "heading_change_deg": heading_change_deg,
            "lateral_offset_m": lateral_offset,
        },
    }
# Example: vehicle driving straight at 10 m/s 
t = np.arange(1, 65) * 0.1 
trajectory = np.column_stack((10.0 * t, np.zeros(64))) # (64, 2) 
result = extract_traj_intent(trajectory, current_speed=10.0) 
print(result)
