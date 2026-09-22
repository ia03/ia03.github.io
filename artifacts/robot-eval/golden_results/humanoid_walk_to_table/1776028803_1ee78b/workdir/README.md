# Task: Humanoid Walk To Table

## Goal
Control the Unitree H1 humanoid to walk to the target marker in front of the
table and remain standing there stably after you exit.

## The scene
- Unitree H1 humanoid, initially at the origin
- Table centered ahead of the robot
- Green circular target marker on the floor near the table

## Success criterion
The hidden grader rebuilds the scene, loads your saved state, runs the sim
forward for a short settle period, and requires:
- the pelvis starts near the origin and makes forward progress through hidden checkpoints
- the pelvis stays within the target radius
- the pelvis remains above a minimum height
- the torso remains upright enough to count as standing

### Success semantics
What must happen:
- The humanoid must make real forward locomotion progress from the canonical start.
- It must reach the target area and remain standing there stably.

What does not count:
- Saving a convenient final pose near the target without real forward progress.
- Directly editing simulator state without corresponding controls.

What is checked over time vs at the end:
- The saved control history is replayed from the canonical start state.
- During replay, the grader checks forward progress through hidden checkpoints and uprightness over time.
- After replay, the grader checks final target proximity and stable standing.

Objective diagnostics:
- The grader reports measurements such as checkpoint hits, target distance, and stable-trace fraction.
- These diagnostics do not change the binary success rule.

### Exact evaluation conditions
- Replay the saved control history from the canonical start state.
- Then run an additional 500 simulation steps of settle.
- Pass iff all of the following are true:
  - The replay has at least `30` control steps.
  - Initial pelvis xy distance from the origin is at most `0.15 m`.
  - During replay, pelvis x reaches all checkpoints: `0.25`, `0.55`, and `0.85`.
  - During replay, the fraction of steps with pelvis z at least `0.75` and torso-up at least `0.8` is at least `0.95`.
  - After settle, pelvis xy distance to the target center `(1.05, 0.0)` is at most `0.18 m`.
  - During settle, minimum pelvis z is at least `0.75`.
  - During settle, minimum torso-up is at least `0.8`.

### Progress score
- The grader also reports `progress_score` in `[0, 1]`.
- `checkpoint_progress = fraction of checkpoints hit among {0.25, 0.55, 0.85}`
- `stability_progress = clamp(stable_trace_fraction / 0.95, 0, 1)`
- `distance_progress = clamp(1 - final_target_distance / 1.05, 0, 1)`
- `settle_posture_progress = 0.5*clamp(min_settle_pelvis_z / 0.75, 0, 1) + 0.5*clamp(min_settle_torso_up / 0.8, 0, 1)`
- `efficiency_progress = clamp(1800 / replay_steps, 0, 1)`
- `progress_score = 0.25*checkpoint_progress + 0.25*stability_progress + 0.25*distance_progress + 0.15*settle_posture_progress + 0.10*efficiency_progress`

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
- A fair baseline is to hold the home posture first before attempting forward motion.
- This task is locomotion-only. You do not need to manipulate the table.
- Save any attempt that gets meaningfully closer to the target while remaining upright.
