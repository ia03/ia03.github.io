# Task: Pick Cup With Wall

## Goal
Pick up the cup from behind the wall, bring it back to the robot side of the
wall, and keep it lifted to **z >= 0.56m**.

## The scene
- Franka Panda arm
- Large table
- Cup at roughly (0.57, 0.10, 0.435)
- Tall wall standing between the robot base and the cup

## Success criterion
The hidden grader rebuilds the scene, loads your saved state, runs the sim
forward, and requires:
- the cup ends up back on the robot side of the wall
- the cup stays lifted above the threshold
- the cup remains in gripper contact during the forward sim

### Success semantics
What must happen:
- The cup must be grasped, lifted, brought back across the wall to the robot side, and still be held stably.

What does not count:
- Knocking, bouncing, or sliding the cup over the wall without a real grasp.
- Pinning the cup against non-finger arm geometry.
- Directly editing simulator state without corresponding controls.

What is checked over time vs at the end:
- The saved control history is replayed from the canonical start state.
- During replay, the grader checks that the cup was actually grasped, lifted, and brought back across the wall.
- After replay, the grader checks the final settled cup position, height, and sustained finger contact.

Objective diagnostics:
- The grader reports measurements such as best replay cup height and best replay cup x-position relative to the wall.
- These diagnostics do not change the binary success rule.

### Exact evaluation conditions
- Replay the saved control history from the canonical start state.
- Then run an additional 500 simulation steps of settle.
- Pass iff all of the following are true:
  - The replay has at least `20` control steps.
  - At some replay step, the cup is in finger contact.
  - At some replay step, the cup reaches `x <= 0.50`.
  - At some replay step, the cup reaches `z >= 0.56`.
  - After settle, the cup final position satisfies `x <= 0.42` and `z >= 0.56`.
  - Finger-cup contact fraction during the 500-step settle is at least `0.5`.

### Progress score
- The grader also reports `progress_score` in `[0, 1]`.
- If the run passes, `progress_score = 1.0`.
- Otherwise:
  - `grasp_progress = 1` if the cup is ever in finger contact during replay, else `0`
  - `return_progress = clamp((0.57 - best_replay_cup_x) / (0.57 - 0.50), 0, 1)`
  - `lift_progress = clamp((best_replay_cup_z - 0.435) / (0.56 - 0.435), 0, 1)`
  - `settle_contact_progress = clamp(settle_contact_fraction / 0.5, 0, 1)`
  - `s1 = grasp_progress`
  - `s2 = grasp_progress * return_progress`
  - `s3 = grasp_progress * return_progress * lift_progress`
  - `s4 = grasp_progress * return_progress * lift_progress * settle_contact_progress`
  - `progress_score = (s1 + s2 + s3 + s4) / 4`

Save with:
```python
sim.save_final_state("/work/final_state.npz")
```

## Hints
- The obvious front approach is obstructed.
- A simple lift behind the wall is not enough. You need retrieval as well.
- Try side approaches or higher approach trajectories.
- Save any promising grasp immediately.
