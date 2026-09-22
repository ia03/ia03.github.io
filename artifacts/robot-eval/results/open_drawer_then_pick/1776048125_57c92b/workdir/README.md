# Task: Open Drawer Then Pick

## Goal
Open the drawer, remove the red block from inside it, and lift the block to
**z >= 0.595m** while the drawer remains open enough to show the retrieval was
real.

## The scene
- Franka Panda arm
- Table
- Roofed cabinet with a sliding drawer and front handle
- Red block initially resting on the drawer tray, so opening the roofed drawer physically presents it through the front opening
- A dedicated drawer actuator named `drawer_motor` is available to control the slide joint directly.

## Success criterion
The hidden grader rebuilds the scene, checks that the drawer was genuinely
opened before the block was lifted out, then runs the sim forward and requires:
- the drawer remains open by at least a threshold
- the block is lifted high enough
- the block is genuinely held during the forward sim

### Success semantics
What must happen:
- The drawer must be opened.
- The red block must then be retrieved through the opened front of the roofed drawer and lifted high enough.
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
  - During replay, the drawer opens to at least `0.05 m` before the block ever reaches `z >= 0.50 m`.
  - After settle, the drawer opening is still at least `0.05 m`.
  - After settle, the block height is at least `0.595 m`.
  - Finger-block contact fraction during the 500-step settle is at least `0.4`.

### Progress score
- The grader also reports `progress_score` in `[0, 1]`.
- `drawer_progress = clamp(max_replay_drawer_open / 0.05, 0, 1)`
- `retrieval_progress = clamp((max_replay_block_z - 0.435) / (0.50 - 0.435), 0, 1)`
- `order_progress = 1` if drawer opening happens before retrieval, else `0`
- `ordered_retrieval_progress = order_progress * retrieval_progress`
- `final_height_progress = clamp((final_block_z_after_settle - 0.43) / (0.595 - 0.43), 0, 1)`
- `hold_progress = clamp(settle_contact_fraction / 0.4, 0, 1)`
- `final_drawer_progress = clamp(final_drawer_open_after_settle / 0.05, 0, 1)`
- `progress_score = 0.22*drawer_progress + 0.28*ordered_retrieval_progress + 0.22*final_height_progress + 0.18*hold_progress + 0.10*final_drawer_progress`

Save with:
```python
sim.save_final_state("/work/final_state.npz")
```

## Hints
- This is a multi-stage task. Solve drawer opening first, then retrieval.
- The drawer uses a slide joint. Inspect joint names with `python sim.py`.
- The drawer also exposes an actuator named `drawer_motor`, so you can open it directly through `data.ctrl`.
- The cabinet stays roofed. Opening the drawer is what presents the block into a reachable state.
- This is still a front-retrieval task: the block has to come out through the opened drawer, not by reaching through the roof.
- Save partial progress whenever you have the drawer open or the block lifted.
