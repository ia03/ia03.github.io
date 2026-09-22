# Task: Open Drawer Then Pick

## Goal
Open the drawer, remove the red block from inside it, and lift the block to
**z >= 0.60m** while the drawer remains open enough to show the retrieval was
real.

## The scene
- Franka Panda arm
- Table
- Cabinet with a sliding drawer and front handle
- Red block initially resting on the cabinet shelf behind the closed drawer front

## Success criterion
The hidden grader rebuilds the scene, checks that the drawer was genuinely
opened before the block was lifted out, then runs the sim forward and requires:
- the drawer remains open by at least a threshold
- the block is lifted high enough
- the block is genuinely held during the forward sim

### Success semantics
What must happen:
- The drawer must be opened.
- The red block must then be removed from the opened drawer and lifted high enough.
- The final held state must remain stable after replay and settle.

What does not count:
- A final pose where the block is outside the drawer without the drawer having actually been opened first.
- Directly editing simulator state without corresponding controls.

What is checked over time vs at the end:
- The saved control history is replayed from the canonical start state.
- During replay, the grader checks stage order: the drawer must open before retrieval and lift.
- After replay, the grader checks final drawer openness, final block height, and stable hold.

Objective diagnostics:
- The grader reports measurements such as maximum drawer opening and maximum block height reached during replay.
- These diagnostics do not change the binary success rule.

### Exact evaluation conditions
- Replay the saved control history from the canonical start state.
- Then run an additional 500 simulation steps of settle.
- Pass iff all of the following are true:
  - The replay has at least `20` control steps.
  - During replay, the drawer opens to at least `0.10 m` before the block ever reaches `z >= 0.50 m`.
  - After settle, the drawer opening is still at least `0.10 m`.
  - After settle, the block height is at least `0.60 m`.
  - Finger-block contact fraction during the 500-step settle is at least `0.4`.

Save with:
```python
sim.save_final_state("/work/final_state.npz")
```

## Hints
- This is a multi-stage task. Solve drawer opening first, then retrieval.
- The drawer uses a slide joint. Inspect joint names with `python sim.py`.
- Save partial progress whenever you have the drawer open or the block lifted.
