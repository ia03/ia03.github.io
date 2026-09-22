# Golden Reference: humanoid_carry_box_to_target

This folder contains a deterministic reference controller for the humanoid carry task.

The reference script:
- runs a small fixed set of candidate controllers
- scores them with the same objective progress components used by the current replay-first grader
- saves the best run to `/work/final_state.npz`

Intended use:

```bash
python harness/run_golden.py --task humanoid_carry_box_to_target
```

The script is deterministic and does not use any randomness or model inference.
