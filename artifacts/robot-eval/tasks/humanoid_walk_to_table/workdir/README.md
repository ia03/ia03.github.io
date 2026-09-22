# Task: Humanoid Walk To Table

## Goal
Control the Unitree H1 humanoid to walk to the target marker in front of the
table and finish there in a controlled crouched/leaning low stance after you exit.

## The scene
- Unitree H1 humanoid, initially at the origin
- Table centered ahead of the robot but closer than the original benchmark layout
- Green circular target marker on the floor just in front of the table
- Reduced gravity (`7.0 m/s^2`) to make replay-first locomotion less brittle

## Success criterion
The hidden grader rebuilds the scene, loads your saved state, runs the sim
forward for a short settle period, and requires:
- the pelvis starts near the origin and makes forward progress through the listed checkpoints
- the pelvis stays within the target radius
- the pelvis remains above a minimum height
- the torso remains above the public torso-up threshold

### Success semantics
What must happen:
- The humanoid must make real forward locomotion progress from the canonical start.
- It must reach the target area and remain there while satisfying the public pelvis-height and torso-up thresholds.

What does not count:
- Saving a convenient final pose near the target without real forward progress.
- Directly editing simulator state without corresponding controls.
- Falling onto the ground near the target.

What is checked over time vs at the end:
- The saved control history is replayed from the canonical start state.
- During replay, the grader checks forward progress through the listed checkpoints and whether the torso/pelvis stay within the allowed low-stance band often enough.
- After replay, the grader checks final target proximity and that the robot does not collapse below the minimum low-stance posture.

Objective diagnostics:
- The grader reports measurements such as checkpoint hits, target distance, and stable-trace fraction.
- These diagnostics do not change the binary success rule.

### Exact evaluation conditions
- Replay the saved control history from the canonical start state.
- Then run an additional 500 simulation steps of settle.
- Pass iff all of the following are true:
  - The replay has at least `30` control steps.
  - Initial pelvis xy distance from the origin is at most `0.15 m`.
  - During replay, pelvis x reaches all checkpoints: `0.18`, `0.34`, and `0.45`.
  - During replay, the fraction of steps with pelvis z at least `0.23` and torso-up at least `0.55` is at least `0.35`.
  - After settle, pelvis xy distance to the target center `(0.58, 0.0)` is at most `0.20 m`.
  - During settle, minimum pelvis z is at least `0.23`.
  - During settle, minimum torso-up is at least `0.55`.

### Progress score
- The grader also reports `progress_score` in `[0, 1]`.
- `checkpoint_progress = fraction of checkpoints hit among {0.18, 0.34, 0.45}`
- `stability_progress = clamp(stable_trace_fraction / 0.35, 0, 1)`
- `distance_progress = 1.0 if final_target_distance <= 0.20 else clamp(1 - (final_target_distance - 0.20) / 0.20, 0, 1)`
- `settle_posture_progress = 0.5*clamp(min_settle_pelvis_z / 0.23, 0, 1) + 0.5*clamp(min_settle_torso_up / 0.55, 0, 1)`
- `efficiency_progress = clamp(1600 / replay_steps, 0, 1)`
- `progress_score = 0.25*checkpoint_progress + 0.25*stability_progress + 0.30*distance_progress + 0.15*settle_posture_progress + 0.05*efficiency_progress`

Save with:
```python
sim.save_final_state("/work/final_state.npz")
```

## Helpful state
- `sim.pelvis_position()`
- `sim.torso_up()`
- `sim.distance_to_target()`
- `sim.home_ctrl()`

## Hints
- Start by understanding the home pose and actuator ordering from `python sim.py`.
- A fair baseline is to hold the home posture first before attempting forward motion, then transition into a controlled low stop near the marker.
- This task is locomotion-only. You do not need to manipulate the table.
- Save any attempt that gets meaningfully closer to the target while remaining upright.
