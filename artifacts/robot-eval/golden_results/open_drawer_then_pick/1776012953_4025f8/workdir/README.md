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

Stage order matters: the block must actually be retrieved from the opened
drawer, not merely appear outside it in a good final pose.
The grader checks what happened over time, so do not save immediately after a
momentary success. Let the open-then-retrieve sequence and final hold play out
briefly before the final save.
Success is judged from replaying your saved control history from the canonical
start state. Directly editing simulator state without corresponding controls
does not count.

Save with:
```python
sim.save_final_state("/work/final_state.npz")
```

## Hints
- This is a multi-stage task. Solve drawer opening first, then retrieval.
- The drawer uses a slide joint. Inspect joint names with `python sim.py`.
- Save partial progress whenever you have the drawer open or the block lifted.
