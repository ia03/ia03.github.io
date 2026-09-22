import numpy as np
import mujoco

from sim import Sim


ARM_DOF = slice(0, 7)


def body_pos_jacobian(sim, body_name):
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jacp = np.zeros((3, sim.model.nv), dtype=np.float64)
    jacr = np.zeros((3, sim.model.nv), dtype=np.float64)
    mujoco.mj_jacBody(sim.model, sim.data, jacp, jacr, bid)
    return jacp[:, :7]


def solve_body_targets(sim, targets, q_init, iters=200, damping=1e-3):
    q = q_init.copy()
    for _ in range(iters):
        sim.data.qpos[:] = q
        mujoco.mj_forward(sim.model, sim.data)
        err_blocks = []
        jac_blocks = []
        for body_name, target_pos in targets:
            bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            err_blocks.append(np.asarray(target_pos, dtype=np.float64) - sim.data.xpos[bid].copy())
            jac_blocks.append(body_pos_jacobian(sim, body_name))
        err = np.concatenate(err_blocks)
        J = np.vstack(jac_blocks)
        if np.linalg.norm(err) < 1e-4:
            break
        dq = np.linalg.solve(J.T @ J + damping * np.eye(7), J.T @ err)
        q[:7] += 0.30 * dq
        for j in range(7):
            lo, hi = sim.model.jnt_range[j]
            q[j] = np.clip(q[j], lo, hi)
    return q


def step_hold(sim, arm_q, finger_ctrl, steps):
    for _ in range(steps):
        sim.data.ctrl[:7] = arm_q[:7]
        sim.data.ctrl[7] = finger_ctrl
        sim.step()


def run():
    sim = Sim()
    q0 = sim.data.qpos.copy()
    cup_pos = sim.cup_position().copy()
    x = cup_pos[0] + 0.04
    yoff = 0.03

    pre_targets = [
        ("left_finger", np.array([x, -yoff, 0.70])),
        ("right_finger", np.array([x, yoff, 0.70])),
    ]
    grasp_targets = [
        ("left_finger", np.array([x, -yoff, 0.44])),
        ("right_finger", np.array([x, yoff, 0.44])),
    ]
    lift_targets = [
        ("left_finger", np.array([x, -yoff, 0.75])),
        ("right_finger", np.array([x, yoff, 0.75])),
    ]

    q_pre = solve_body_targets(sim, pre_targets, q0)
    q_grasp = solve_body_targets(sim, grasp_targets, q_pre)
    q_lift = solve_body_targets(sim, lift_targets, q_grasp)

    # Approach with the gripper open.
    step_hold(sim, q_pre, 255.0, 260)
    for t in range(180):
        a = (t + 1) / 180.0
        q = (1 - a) * q_pre + a * q_grasp
        step_hold(sim, q, 255.0, 1)

    # Let the open gripper settle around the cup, then close.
    step_hold(sim, q_grasp, 255.0, 60)
    step_hold(sim, q_grasp, 0.0, 260)

    # Lift while holding closed.
    for t in range(360):
        a = (t + 1) / 360.0
        q = (1 - a) * q_grasp + a * q_lift
        step_hold(sim, q, 0.0, 1)

    # Stable hold before saving.
    step_hold(sim, q_lift, 0.0, 240)

    hand_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    print("time", sim.data.time)
    print("cup", sim.cup_position())
    print("hand", sim.data.xpos[hand_bid])
    print("qpos", sim.data.qpos[:9])
    sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    run()
