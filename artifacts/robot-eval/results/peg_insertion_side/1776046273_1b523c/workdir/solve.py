import time
import numpy as np
import mujoco

import sim


ARM_DOF = 7
GRIPPER_ACT = 7


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def body_xaxis_alignment(model, data, body_name="peg"):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    xmat = data.xmat[bid].reshape(3, 3)
    return float(abs(xmat[0, 0]))


def mat_from_body(data, body_id):
    return data.xmat[body_id].reshape(3, 3).copy()


def solve_arm_ik(
    s,
    target_pos,
    target_rot=None,
    q_seed=None,
    iters=10,
    damping=1e-3,
    step=0.25,
):
    """Iterative damped least-squares IK for the hand body position."""
    model, data = s.model, s.data
    hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    if q_seed is None:
        q = data.qpos[:ARM_DOF].copy()
    else:
        q = np.array(q_seed, dtype=float).copy()

    for _ in range(iters):
        data.qpos[:ARM_DOF] = q
        mujoco.mj_forward(model, data)
        pos = data.xpos[hand_bid].copy()
        err = np.asarray(target_pos) - pos
        if np.linalg.norm(err) < 1e-4:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, hand_bid)
        J = jacp[:, :ARM_DOF]
        H = J @ J.T + damping * np.eye(3)
        dq = J.T @ np.linalg.solve(H, err)
        q = q + step * dq
        q = clamp(q, model.jnt_range[:ARM_DOF, 0], model.jnt_range[:ARM_DOF, 1])

    return q


def drive_hand(
    s,
    target_pos,
    target_rot,
    n_steps,
    gripper=255.0,
    ik_iters=30,
    ik_step=0.20,
):
    """Move the hand toward a target Cartesian point using joint-position targets."""
    hand_bid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    q_target = solve_arm_ik(
        s,
        target_pos,
        target_rot,
        q_seed=s.data.qpos[:ARM_DOF].copy(),
        iters=ik_iters,
        step=ik_step,
    )
    for _ in range(n_steps):
        s.data.ctrl[:ARM_DOF] = q_target
        s.data.ctrl[GRIPPER_ACT] = gripper
        s.step()
    return s.data.xpos[hand_bid].copy()


def print_state(s, label):
    bid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    hid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    pos = s.data.xpos[bid].copy()
    hpos = s.data.xpos[hid].copy()
    hmat = s.data.xmat[hid].reshape(3, 3)
    align = body_xaxis_alignment(s.model, s.data, "peg")
    print(
        f"{label}: t={s.data.time:.3f} peg_pos={np.array2string(pos, precision=4)} "
        f"hand_pos={np.array2string(hpos, precision=4)} hand_x={np.array2string(hmat[:,0], precision=3)} "
        f"align={align:.3f} ctrl7={s.data.ctrl[GRIPPER_ACT]:.1f}"
    )


def main():
    s = sim.Sim()
    task_rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    t0 = time.time()
    print_state(s, "start")

    # Phase 1: move to a pre-contact pose just behind the tab.
    drive_hand(s, target_pos=[0.395, -0.100, 0.522], target_rot=task_rot, n_steps=240, gripper=255.0)
    print_state(s, "after_precontact")
    # Save an early plausible attempt so the file exists even if later refinement fails.
    s.save_final_state("/work/final_state.npz")

    # Phase 2: clamp the tab a bit before pushing.
    drive_hand(s, target_pos=[0.410, -0.100, 0.520], target_rot=task_rot, n_steps=240, gripper=120.0)
    print_state(s, "after_clamp")

    # Phase 3: push in +x toward the slot center.
    push_targets = np.linspace(0.412, 0.610, 120)
    for x in push_targets:
        drive_hand(s, target_pos=[float(x), -0.100, 0.520], target_rot=task_rot, n_steps=8, gripper=120.0)
    print_state(s, "after_push")

    # Phase 4: keep holding the line while the peg settles deep in the slot.
    drive_hand(s, target_pos=[0.605, -0.100, 0.520], target_rot=task_rot, n_steps=300, gripper=120.0)
    print_state(s, "after_hold")

    # Let contacts settle before saving the final state.
    for _ in range(500):
        s.data.ctrl[:ARM_DOF] = s.data.qpos[:ARM_DOF]
        s.data.ctrl[GRIPPER_ACT] = 120.0
        s.step()

    print_state(s, "final_settled")
    s.save_final_state("/work/final_state.npz")
    print(f"saved /work/final_state.npz in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
