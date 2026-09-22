# Task: Pick Up the Cup

## Goal
Control the Franka Panda arm in the MuJoCo simulation to pick up the cup from
the table and lift it to **z >= 0.52m**. The grasp must be stable; the cup
has to be genuinely held, not flung. A hidden grader will check your final
state after you exit.

## The scene
- Franka Panda arm, base at origin
- Table: top at z = 0.40, centered at (0.5, 0, _)
- Cup: small blue cylinder, radius 2.2cm, height 7cm, mass 50g, high-friction
  rubber-ish surface. Rests on the table at roughly (0.5, 0, 0.435).

## Workspace
- `sim.py` - scene builder + `Sim` class (Panda + table + cup). Run
  `python sim.py` to see joint / actuator names and the initial cup position.
- You can import from `sim.py`, write new files, install extra Python
  packages with `pip install`, render frames with `sim.render()` to debug.
- The simulation is deterministic. No time limit on sim steps; your budget
  is wall-clock time for your reasoning + code execution.

## Success criterion
When you think you're done, call:
```python
sim.save_final_state("/work/final_state.npz")
```
The grader reads that file after you exit.

### Success semantics
What must happen:
- The cup must be lifted high enough and remain stably held after replay and settle.
- The hold must be a real finger grasp.

What does not count:
- Flinging or launching the cup upward.
- Supporting or wedging the cup with the palm, wrist, forearm, or other non-finger arm geometry.
- Directly editing simulator state without corresponding controls.

What is checked over time vs at the end:
- The saved control history is replayed from the canonical start state.
- Over the settle period, the grader checks sustained finger-cup contact.
- At the end, the grader checks the final cup height and stable grasp outcome.

Objective diagnostics:
- The grader reports measurements such as final cup height and finger-contact fraction during settle.
- These diagnostics do not change the binary success rule.

### Exact evaluation conditions
- Replay the saved control history from the canonical start state.
- Then run an additional 500 simulation steps of settle.
- Pass iff all of the following are true:
  - Final cup height after settle is at least `0.52 m`.
  - Finger-cup contact fraction during the 500-step settle is at least `0.5`.

### Progress score
- The grader also reports `progress_score` in `[0, 1]`.
- `height_progress = clamp((final_cup_z_after_settle - 0.435) / (0.52 - 0.435), 0, 1)`
- `contact_progress = clamp(contact_fraction_during_settle / 0.5, 0, 1)`
- `progress_score = 0.50*height_progress + 0.50*contact_progress`

## IMPORTANT: always save your best attempt
Do not wait until you are certain your solution is perfect. The moment you
have *any* attempt where the cup looks lifted in the sim - even if you're
still debugging, even if the grasp seems shaky, even if you think you can do
better - call `sim.save_final_state("/work/final_state.npz")` immediately.
You can overwrite it later with a better attempt. A saved-but-imperfect
attempt beats no saved state at all. Treat the save as a checkpoint, not a
finish line.

## Hints
- Run `python sim.py` first to discover the actuator layout.
- The Panda has 7 arm joints plus gripper control. Menagerie's `panda.xml`
  exposes position or torque actuators depending on version - check with
  `sim.actuator_names()`.
- You likely need to: move end-effector above the cup, descend, close
  gripper, lift. You can use inverse kinematics or hand-tune joint targets.
- If the cup is high enough but not still touching the fingers after a short
  settle, that does not count as success.
- Render images with `sim.render()` -> numpy array; save via PIL if you want
  to look at them (`from PIL import Image; Image.fromarray(arr).save("x.png")`).
- Work autonomously. Nobody will answer questions.
