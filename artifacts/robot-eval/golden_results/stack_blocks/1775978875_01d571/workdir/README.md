# Task: Stack Blocks

## Goal
Stack the red block on top of the green block and leave the stack standing
stably after you exit.

## The scene
- Franka Panda arm
- Table at z = 0.40
- Red block at roughly (0.48, -0.09, 0.425)
- Green block at roughly (0.63, 0.08, 0.425)

## Success criterion
The hidden grader rebuilds the scene, runs the sim forward, and requires:
- the red block center is above the green block center
- the xy alignment is tight enough to count as stacked
- the stack remains stable after settling

### Success semantics
What must happen:
- The red block must end up stably stacked on top of the green block.
- The final stack must be horizontally aligned enough and at the correct stacked height.

What does not count:
- A leaning or sliding arrangement that does not remain stable after settling.
- Directly editing simulator state without corresponding controls.

What is checked over time vs at the end:
- The saved control history is replayed from the canonical start state.
- This task is judged by the final physical result after replay and settle, not by any specific motion sequence.

Objective diagnostics:
- The grader reports measurements such as final xy alignment error, stack-height offset, and replay length.
- These diagnostics are for analysis only. Binary success still depends only on ending with a genuinely stable stack.

### Exact evaluation conditions
- Replay the saved control history from the canonical start state.
- Then run an additional 500 simulation steps of settle.
- Let `dx, dy, dz` be red-block minus green-block center offsets after settle.
- Pass iff all of the following are true:
  - `|dx| <= 0.018`.
  - `|dy| <= 0.018`.
  - `0.040 <= dz <= 0.070`.

### Progress score
- The grader also reports `progress_score` in `[0, 1]`.
- `completion_progress = 1` if the final settled stack passes the binary success rule, else `0`
- `xy_progress = clamp(1 - max(|dx|, |dy|) / 0.018, 0, 1)`
- `z_progress = clamp(1 - |dz - 0.055| / 0.015, 0, 1)`
- `rotation_progress = clamp((final_block_yaw_alignment - 0.90) / 0.10, 0, 1)`
- `efficiency_progress = clamp(900 / replay_steps, 0, 1)`
- `progress_score = 0.35*completion_progress + 0.25*xy_progress + 0.20*z_progress + 0.15*rotation_progress + 0.05*efficiency_progress`

Save with:
```python
sim.save_final_state("/work/final_state.npz")
```

## Hints
- A successful grasp is only the midpoint. Placement stability is what matters.
- Release cleanly and let the stack settle before saving.
