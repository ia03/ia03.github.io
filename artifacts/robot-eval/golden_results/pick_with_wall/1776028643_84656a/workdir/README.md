# Task: Pick Cup With Wall

## Goal
Pick up the cup from behind the wall, bring it back to the robot side of the
wall, and keep it lifted to **z >= 0.56m**.

## The scene
- Franka Panda arm
- Large table
- Cup at roughly (0.55, 0.15, 0.435)
- A low wall standing between the robot base and the cup. It still blocks the direct shortest path, but it is lower and narrower so a replay-first over-the-top retrieval is physically viable without relying on borderline contact hacks.

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
  - At some replay step, the cup reaches `x <= 0.48`.
  - At some replay step, the cup reaches `z >= 0.56`.
  - After settle, the cup final position satisfies `x <= 0.42` and `z >= 0.56`.
  - Finger-cup contact fraction during the 500-step settle is at least `0.5`.

### Progress score
- The grader also reports `progress_score` in `[0, 1]`.
- `grasp_progress = 1` if the cup is ever in finger contact during replay, else `0`
- `return_progress = clamp((0.55 - best_replay_cup_x) / (0.55 - 0.48), 0, 1)`
- `lift_progress = clamp((best_replay_cup_z - 0.435) / (0.56 - 0.435), 0, 1)`
- `final_return_progress = clamp((0.55 - final_cup_x_after_settle) / (0.55 - 0.42), 0, 1)`
- `final_height_progress = clamp((final_cup_z_after_settle - 0.435) / (0.56 - 0.435), 0, 1)`
- `settle_contact_progress = clamp(settle_contact_fraction / 0.5, 0, 1)`
- `efficiency_progress = clamp(1200 / replay_steps, 0, 1)`
- `progress_score = 0.05*grasp_progress + 0.15*return_progress + 0.10*lift_progress + 0.10*final_return_progress + 0.25*final_height_progress + 0.25*settle_contact_progress + 0.10*efficiency_progress`

Save with:
```python
sim.save_final_state("/work/final_state.npz")
```

## Hints
- The obvious front approach is obstructed.
- A simple lift behind the wall is not enough. You need retrieval as well.
- Try side approaches or higher approach trajectories.
- Save any promising grasp immediately.
