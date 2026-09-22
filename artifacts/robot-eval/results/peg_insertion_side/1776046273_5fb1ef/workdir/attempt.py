import time
import numpy as np
import mujoco

import sim


ARM_QPOS_SLICE = slice(0, 7)
FINGER_CTRL_IDX = 7


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def hand_body_id(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")


def solve_ik_to_pos(sim_obj, target_pos, q_seed, iters=80, alpha=0.35, damping=1e-4):
    q = q_seed.copy()
    data = sim_obj.data
    model = sim_obj.model
    hid = hand_body_id(model)
    q_full = data.qpos.copy()
    q_full[ARM_QPOS_SLICE] = q
    q_full[7:9] = 0.0
    q_full[9:] = sim_obj._initial_qpos[9:]
    data.qpos[:] = q_full
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    for _ in range(iters):
        pos = data.xpos[hid].copy()
        err = target_pos - pos
        if np.linalg.norm(err) < 1e-4:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, hid)
        J = jacp[:, :7]
        A = J @ J.T + damping * np.eye(3)
        dq = J.T @ np.linalg.solve(A, err)
        q = q + alpha * dq
        q = clamp(q, model.jnt_range[:7, 0], model.jnt_range[:7, 1])
        q_full[ARM_QPOS_SLICE] = q
        data.qpos[:] = q_full
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
    return q


def step_hold(sim_obj, q_arm, n_steps, finger_ctrl=0.0):
    ctrl = np.zeros(sim_obj.model.nu)
    ctrl[:7] = q_arm
    ctrl[FINGER_CTRL_IDX] = finger_ctrl
    for _ in range(n_steps):
        sim_obj.data.ctrl[:] = ctrl
        sim_obj.step(1)


def main():
    s = sim.Sim()
    model = s.model
    q = s.data.qpos[:7].copy()

    # A short safe approach above and left of the peg, then behind the tab,
    # push along +x, and finally retreat upward.
    waypoints = [
        (np.array([0.32, -0.06, 0.78]), 90),
        (np.array([0.38, -0.10, 0.66]), 80),
        (np.array([0.40, -0.102, 0.58]), 70),
        (np.array([0.42, -0.102, 0.53]), 70),
        (np.array([0.45, -0.102, 0.52]), 70),
        (np.array([0.49, -0.102, 0.52]), 70),
    ]

    t0 = time.time()
    for i, (target, steps) in enumerate(waypoints):
        q = solve_ik_to_pos(s, target, q, iters=120, alpha=0.35, damping=1e-4)
        finger_ctrl = 0.0
        step_hold(s, q, steps, finger_ctrl=finger_ctrl)
        print(
            f"stage {i}: target={target}, hand={s.data.xpos[hand_body_id(model)]}, "
            f"peg={s.peg_position()}, time={s.data.time:.3f}, elapsed={time.time()-t0:.1f}s"
        )

    # Let the system settle a bit with the arm held at the best partial insertion pose.
    step_hold(s, q, 160, finger_ctrl=0.0)
    print("final hand", s.data.xpos[hand_body_id(model)])
    print("final peg", s.peg_position())
    print("final q", s.data.qpos[:7])
    s.save_final_state("/work/final_state.npz")
    print("saved /work/final_state.npz")


if __name__ == "__main__":
    main()
