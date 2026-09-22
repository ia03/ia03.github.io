# Task: Humanoid Walk To Table Then Touch Marker

## Goal
Control the Unitree H1 humanoid to walk to the table and touch the red marker
above the front edge of the tabletop while staying roughly upright enough to
count as a controlled approach rather than a full collapse.

## Success criterion
The hidden grader rebuilds the scene, loads your saved state, runs the sim
forward, and requires:
- the pelvis starts near the origin and makes forward progress through hidden checkpoints
- the pelvis remains within the floor target radius near the table
- the pelvis stays above a minimum height
- the torso stays upright enough to count as a controlled crouched or leaning touch
- at least one elbow remains within the touch radius of the tabletop marker

### Success semantics
What must happen:
- The humanoid must walk from the canonical start toward the table.
- It must remain roughly upright enough to count as a controlled crouched or
  leaning touch.
- At least one elbow must reach the tabletop marker while the humanoid is in the correct target area.

What does not count:
- Reaching the marker from a convenient final pose without real locomotion progress from the start.
- Directly editing simulator state without corresponding controls.

What is checked over time vs at the end:
- The saved control history is replayed from the canonical start state.
- During replay, the grader checks forward progress, rough uprightness, and best
  marker-touch distance over time.
- After replay, the grader checks final target proximity and whether the robot
  stays above the minimum crouched/upright thresholds during settle.

Objective diagnostics:
- The grader reports measurements such as checkpoint hits, stable-trace fraction, target distance, and best touch distance.
- These diagnostics do not change the binary success rule.

### Exact evaluation conditions
- Replay the saved control history from the canonical start state.
- Then run an additional 500 simulation steps of settle.
- Pass iff all of the following are true:
  - The replay has at least `30` control steps.
  - Initial pelvis xy distance from the origin is at most `0.15 m`.
  - During replay, pelvis x reaches all checkpoints: `0.25`, `0.55`, and `0.80`.
  - During replay, the fraction of steps with pelvis z at least `0.23` and torso-up at least `0.55` is at least `0.35`.
  - During replay, at least one elbow comes within `0.12 m` of the touch point `(1.18, 0.0, 0.83)`.
  - After settle, pelvis xy distance to the target center `(1.00, 0.0)` is at most `0.45 m`.
  - During settle, minimum pelvis z is at least `0.23`.
  - During settle, minimum torso-up is at least `0.55`.
  - During settle, at least one elbow still reaches within `0.12 m` of the touch point at some point.

### Progress score
- The grader also reports `progress_score` in `[0, 1]`.
- `checkpoint_progress = fraction of checkpoints hit among {0.25, 0.55, 0.80}`
- `touch_progress = 1.0` if the best observed touch distance is within the public touch radius `0.12`, otherwise it decays linearly with distance beyond that radius
- `stability_progress = clamp(stable_trace_fraction / 0.35, 0, 1)`
- `distance_progress = 1.0` if final pelvis target distance is within the public target radius `0.45`, otherwise it decays linearly with distance beyond that radius
- `settle_posture_progress = 0.5*clamp(min_settle_pelvis_z / 0.23, 0, 1) + 0.5*clamp(min_settle_torso_up / 0.55, 0, 1)`
- `efficiency_progress = clamp(2200 / replay_steps, 0, 1)`
- `progress_score = 0.15*checkpoint_progress + 0.30*touch_progress + 0.20*stability_progress + 0.15*distance_progress + 0.10*settle_posture_progress + 0.10*efficiency_progress`

Save with:
```python
sim.save_final_state("/work/final_state.npz")
```

## Helpful state
- `sim.pelvis_position()`
- `sim.torso_up()`
- `sim.distance_to_target()`
- `sim.distance_to_touch()`
- `sim.home_ctrl()`

## Hints
- This is locomotion plus upper-body reaching.
- Stabilize standing first.
- The touch marker is on the table side closest to the robot.
