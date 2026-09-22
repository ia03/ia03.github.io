# Task: Humanoid Carry Box To Target

## Goal
Control the Unitree H1 humanoid to take the large yellow box off the pedestal
and transport it to the floor target while remaining upright.

## Important note
This H1 asset has no hands or grippers. The intended strategy is to cradle or
pin the box against the torso and forearms while walking. That makes this a
fair transport task for this specific model, not a dexterous grasping task.

## Success criterion
The hidden grader rebuilds the scene, loads your saved state, runs the sim
forward, and requires:
- the humanoid and box both make forward progress through hidden checkpoints
- the pelvis ends within the target radius
- the box ends within the target radius
- the box stays above a minimum height
- the torso stays upright and the pelvis stays standing-height
- the box remains close enough to the pelvis to count as being carried

### Success semantics
What must happen:
- The humanoid must take the box off the pedestal and transport it toward the target area.
- The box must stay close enough to the humanoid to count as carried.
- The humanoid must remain upright enough to count as standing.

What does not count:
- The box arriving near the target independently of the humanoid.
- A final pose where the humanoid is near the target but the box was not truly carried there.
- Directly editing simulator state without corresponding controls.

What is checked over time vs at the end:
- The saved control history is replayed from the canonical start state.
- During replay, the grader checks forward progress of both the humanoid and box, carrying proximity, and stability over time.
- After replay, the grader checks final target proximity for both the humanoid and the box and final stable standing.

Objective diagnostics:
- The grader reports measurements such as checkpoint hits, pelvis distance to target, box distance to target, and box-to-pelvis distance.
- These diagnostics do not change the binary success rule.

### Exact evaluation conditions
- Replay the saved control history from the canonical start state.
- Then run an additional 500 simulation steps of settle.
- Pass iff all of the following are true:
  - The replay has at least `30` control steps.
  - Initial pelvis xy distance from the origin is at most `0.15 m`.
  - During replay, pelvis x reaches all checkpoints: `0.35`, `0.65`, and `0.95`.
  - During replay, box x reaches all checkpoints minus `0.05`, i.e. `0.30`, `0.60`, and `0.90`.
  - During replay, the fraction of steps simultaneously satisfying all of the following is at least `0.90`:
    - pelvis z at least `0.74`
    - torso-up at least `0.78`
    - box z at least `0.72`
    - box-to-pelvis distance at most `0.42 m`
  - After settle, pelvis xy distance to the target center `(1.15, 0.0)` is at most `0.22 m`.
  - After settle, box xy distance to the target center `(1.15, 0.0)` is at most `0.22 m`.
  - During settle, minimum pelvis z is at least `0.74`.
  - During settle, minimum torso-up is at least `0.78`.
  - During settle, minimum box z is at least `0.72`.
  - During settle, maximum box-to-pelvis distance is at most `0.42 m`.

### Progress score
- The grader also reports `progress_score` in `[0, 1]`.
- `pelvis_checkpoint_progress = fraction of pelvis checkpoints hit among {0.35, 0.65, 0.95}`
- `box_checkpoint_progress = fraction of box checkpoints hit among {0.30, 0.60, 0.90}`
- `stability_progress = clamp(stable_trace_fraction / 0.90, 0, 1)`
- `pelvis_distance_progress = clamp(1 - final_pelvis_target_distance / 1.15, 0, 1)`
- `box_distance_progress = clamp(1 - final_box_target_distance / 0.97, 0, 1)`
- `carry_proximity_progress = clamp(1 - max(0, max_settle_box_to_pelvis - 0.42) / 0.42, 0, 1)`
- `box_height_progress = clamp(min_settle_box_z / 0.72, 0, 1)`
- `efficiency_progress = clamp(2600 / replay_steps, 0, 1)`
- `progress_score = 0.15*box_distance_progress + 0.30*carry_proximity_progress + 0.25*box_height_progress + 0.10*stability_progress + 0.03*box_checkpoint_progress + 0.02*pelvis_distance_progress + 0.00*pelvis_checkpoint_progress + 0.15*efficiency_progress`

Save with:
```python
sim.save_final_state("/work/final_state.npz")
```

## Helpful state
- `sim.pelvis_position()`
- `sim.box_position()`
- `sim.distance_to_target()`
- `sim.box_distance_to_target()`
- `sim.box_distance_to_pelvis()`
- `sim.home_ctrl()`
