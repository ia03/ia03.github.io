# Task: Peg Insertion Side

## Goal
Insert the orange peg through the side slot in the vertical fixture.

## The scene
- Franka Panda arm
- Table
- Free peg lying on the table
- Vertical fixture with a narrow side slot

## Success criterion
The hidden grader rebuilds the scene, loads your final state, runs the sim
forward, and requires the peg to remain inserted deep enough in the slot with
the correct approximate orientation.

### Success semantics
What must happen:
- The peg must be inserted deeply enough into the side slot.
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
  - Peg center `x >= 0.695`.
  - Peg center `|y - 0.08| <= 0.02`.
  - Peg center `|z - 0.48| <= 0.02`.
  - Absolute peg local x-axis alignment with world x-axis is at least `0.85`.

### Progress score
- The grader also reports `progress_score` in `[0, 1]`.
- If the run passes, `progress_score = 1.0`.
- Otherwise:
  - `insertion_progress = clamp((final_peg_x - 0.47) / (0.695 - 0.47), 0, 1)`
  - `y_progress = clamp(1 - |final_peg_y - 0.08| / 0.02, 0, 1)`
  - `z_progress = clamp(1 - |final_peg_z - 0.48| / 0.02, 0, 1)`
  - `alignment_progress = clamp(|final_peg_x_axis_alignment| / 0.85, 0, 1)`
  - `s1 = insertion_progress`
  - `s2 = insertion_progress * y_progress`
  - `s3 = insertion_progress * y_progress * z_progress`
  - `s4 = insertion_progress * y_progress * z_progress * alignment_progress`
  - `progress_score = (s1 + s2 + s3 + s4) / 4`

Save with:
```python
sim.save_final_state("/work/final_state.npz")
```

## Hints
- This task is about alignment, not just grasping.
- You may need to reason about peg orientation and insertion direction.
- Save intermediate successes aggressively.
