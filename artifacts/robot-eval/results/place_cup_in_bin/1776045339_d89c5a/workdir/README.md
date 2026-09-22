# Task: Place Cup In Bin

## Goal
Pick up the cup from the table and place it inside the bin. The cup must end up
resting stably inside the bin after you exit.

## The scene
- Franka Panda arm, base at origin
- Large table with top at z = 0.40
- Cup at roughly (0.48, -0.12, 0.435)
- Open-top bin at roughly (0.70, 0.14, 0.40)

## Success criterion
The hidden grader rebuilds the scene, loads your saved final state, runs the sim
forward, and checks that:
- the cup is genuinely picked up before being placed
- the cup stays inside the bin bounds
- the cup's symmetry axis remains vertical enough to count as placed
- the cup is no longer being held by the gripper

### Success semantics
What must happen:
- The cup must be picked up, transported into the bin, released, and left resting stably inside the bin.
- The final cup axis must still count as vertical placement. Because the simulated cup is a symmetric cylinder, either sign of the local z-axis is acceptable as long as the cylinder axis is vertical.

What does not count:
- Leaving the cup supported by the gripper at the end.
- Resting the cup on the bin edge or wall without a valid final placement inside.
- Directly editing simulator state without corresponding controls.

What is checked over time vs at the end:
- The saved control history is replayed from the canonical start state.
- During replay, the grader checks that the cup was grasped, lifted, and then released.
- After replay, the grader checks the final settled cup position, cylinder-axis verticality, and lack of continued gripper hold.

Objective diagnostics:
- The grader reports measurements such as best lift height, final bin-center error, and final cup axis verticality.
- These diagnostics do not change the binary success rule.

### Exact evaluation conditions
- Replay the saved control history from the canonical start state.
- Then run an additional 500 simulation steps of settle.
- Pass iff all of the following are true:
  - The replay has at least `20` control steps.
  - At some replay step, the cup is in finger contact.
  - At some replay step, the cup reaches `z >= 0.50`.
  - Over the last 10 replay steps, finger-cup contact fraction is less than `0.5`.
  - After settle, finger-cup contact fraction is less than `0.1`.
  - After settle, the cup center is inside the bin interior with `|x - 0.70| <= 0.046` and `|y - 0.14| <= 0.046`.
  - After settle, the cup center height is at least `0.42 m`.
  - After settle, the cup axis verticality satisfies `abs(local cup z-axis component) >= 0.85`.

### Progress score
- The grader also reports `progress_score` in `[0, 1]`.
- `grasp_progress = 1` if the cup is ever in finger contact during replay, else `0`
- `lift_progress = clamp((best_replay_cup_z - 0.435) / (0.50 - 0.435), 0, 1)`
- `inside_bin_progress = 1` if the final settled cup center is inside the valid bin interior, else `0`
- `placement_progress = 1` when the final settled cup center is already inside the valid bin interior, else `clamp(1 - final_bin_center_distance / initial_bin_center_distance, 0, 1)`
- `release_progress = clamp(1 - settle_contact_fraction / 0.1, 0, 1)`
- `upright_progress = clamp(abs(final_cup_up_z) / 0.85, 0, 1)`
- `progress_score = 0.05*grasp_progress + 0.10*lift_progress + 0.40*inside_bin_progress + 0.10*placement_progress + 0.10*release_progress + 0.25*upright_progress`

Save with:
```python
sim.save_final_state("/work/final_state.npz")
```

## Hints
- Run `python sim.py` first.
- A direct lift is not enough. You need transport plus release.
- Plan for final orientation as well as position. A cup that lands sideways in
  the bin is not a success.
- Save your best attempt early and overwrite it later if you improve it.
