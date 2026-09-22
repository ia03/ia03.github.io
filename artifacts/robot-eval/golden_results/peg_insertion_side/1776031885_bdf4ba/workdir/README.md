# Task: Peg Insertion Side

## Goal
Insert the orange peg through the side slot in the vertical fixture.

## The scene
- Franka Panda arm
- Table
- Peg resting on a short guide shelf near the slot
- Vertical fixture with a broader side slot centered on the guide shelf lane

## Success criterion
The hidden grader rebuilds the scene, loads your final state, runs the sim
forward, and requires the peg to remain inserted deep enough in the slot with
the correct approximate orientation.

### Success semantics
What must happen:
- The peg must be driven from the guide shelf into the side slot deeply enough.
- The final inserted pose must have the required approximate orientation.

What does not count:
- A transient poke that does not remain inserted after settling.
- A final pose where the peg is in the slot region but at the wrong angle or depth.
- Directly editing simulator state without corresponding controls.

What is checked over time vs at the end:
- The saved control history is replayed from the canonical start state.
- The final settled replay state is used to judge insertion depth and orientation.

Objective diagnostics:
- The grader reports measurements such as insertion depth and final peg alignment.
- These diagnostics do not change the binary success rule.

### Exact evaluation conditions
- Replay the saved control history from the canonical start state.
- Then run an additional 500 simulation steps of settle.
- Pass iff all of the following are true in the final settled state:
  - Peg center `x >= 0.55`.
  - Peg center `|y - (-0.10)| <= 0.025`.
  - Peg center `|z - 0.48| <= 0.020`.
  - Absolute peg local x-axis alignment with world x-axis is at least `0.45`.

### Progress score
- The grader also reports `progress_score` in `[0, 1]`.
- `insertion_progress = clamp((final_peg_x - 0.505) / (0.55 - 0.505), 0, 1)`
- `y_progress = clamp(1 - |final_peg_y - (-0.10)| / 0.025, 0, 1)`
- `z_progress = clamp(1 - |final_peg_z - 0.48| / 0.020, 0, 1)`
- `alignment_progress = clamp(|final_peg_x_axis_alignment| / 0.45, 0, 1)`
- `efficiency_progress = clamp(1200 / replay_steps, 0, 1)`
- `progress_score = 0.45*insertion_progress + 0.15*y_progress + 0.15*z_progress + 0.20*alignment_progress + 0.05*efficiency_progress`

Save with:
```python
sim.save_final_state("/work/final_state.npz")
```

## Hints
- This task is about controlled insertion from the guide shelf, not long transport.
- You may not need a full table-to-slot pick and place if you can maintain the correct lane and orientation.
- Save intermediate successes aggressively.
