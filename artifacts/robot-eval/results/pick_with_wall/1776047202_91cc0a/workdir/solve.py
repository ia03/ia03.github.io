import numpy as np
import mujoco

from sim import Sim, SUCCESS_CUP_Z


def quat_from_mat(mat9):
    q = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(q, mat9)
    return q


def quat_mul(q1, q2):
    out = np.zeros(4, dtype=float)
    mujoco.mju_mulQuat(out, q1, q2)
    return out


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def clip_ctrl_to_range(model, ctrl):
    out = ctrl.copy()
    for i in range(model.nu):
        lo, hi = model.actuator_ctrlrange[i]
        out[i] = float(np.clip(out[i], lo, hi))
    return out


def ik_step_pos(sim, body_id, pos_des, arm_dofs=7, kp_pos=10.0, damping=0.06):
    m, d = sim.model, sim.data

    jacp = np.zeros((3, m.nv), dtype=float)
    jacr = np.zeros((3, m.nv), dtype=float)
    mujoco.mj_jacBody(m, d, jacp, jacr, body_id)
    J = jacp[:, :arm_dofs]

    pos_cur = np.array(d.xpos[body_id], dtype=float)
    pos_err = pos_des - pos_cur
    v = np.clip(kp_pos * pos_err, -1.5, 1.5)

    A = J @ J.T + (damping**2) * np.eye(3)
    dq = J.T @ np.linalg.solve(A, v)

    dt = float(m.opt.timestep)
    q_arm = d.qpos[:arm_dofs].copy()
    q_ref = q_arm + dq * dt
    return q_ref


def ik_step_midpoint(sim, body_a_id, body_b_id, pos_des, arm_dofs=7, kp_pos=10.0, damping=0.06):
    m, d = sim.model, sim.data

    jacp_a = np.zeros((3, m.nv), dtype=float)
    jacr_a = np.zeros((3, m.nv), dtype=float)
    mujoco.mj_jacBody(m, d, jacp_a, jacr_a, body_a_id)

    jacp_b = np.zeros((3, m.nv), dtype=float)
    jacr_b = np.zeros((3, m.nv), dtype=float)
    mujoco.mj_jacBody(m, d, jacp_b, jacr_b, body_b_id)

    J = 0.5 * (jacp_a[:, :arm_dofs] + jacp_b[:, :arm_dofs])

    pos_a = np.array(d.xpos[body_a_id], dtype=float)
    pos_b = np.array(d.xpos[body_b_id], dtype=float)
    pos_mid = 0.5 * (pos_a + pos_b)
    pos_err = pos_des - pos_mid
    v = np.clip(kp_pos * pos_err, -1.5, 1.5)

    A = J @ J.T + (damping**2) * np.eye(3)
    dq = J.T @ np.linalg.solve(A, v)

    dt = float(m.opt.timestep)
    q_arm = d.qpos[:arm_dofs].copy()
    q_ref = q_arm + dq * dt
    return q_ref


def ik_step_pose(sim, body_id, pos_des, quat_des, arm_dofs=7, kp_pos=8.0, kp_ori=3.0, damping=0.08):
    m, d = sim.model, sim.data

    jacp = np.zeros((3, m.nv), dtype=float)
    jacr = np.zeros((3, m.nv), dtype=float)
    mujoco.mj_jacBody(m, d, jacp, jacr, body_id)
    J = np.vstack([jacp[:, :arm_dofs], jacr[:, :arm_dofs]])

    pos_cur = np.array(d.xpos[body_id], dtype=float)
    pos_err = pos_des - pos_cur

    quat_cur = quat_from_mat(d.xmat[body_id])
    q_err = quat_mul(quat_des, quat_conj(quat_cur))
    if q_err[0] < 0:
        q_err *= -1
    ori_err = 2.0 * q_err[1:]

    v = np.concatenate([kp_pos * pos_err, kp_ori * ori_err])
    v = np.clip(v, -1.5, 1.5)

    A = J @ J.T + (damping**2) * np.eye(6)
    dq = J.T @ np.linalg.solve(A, v)

    dt = float(m.opt.timestep)
    q_arm = d.qpos[:arm_dofs].copy()
    q_ref = q_arm + dq * dt
    return q_ref


def run_phase(sim, target, pos_des, steps, gripper_cmd, settle=False, log_every=200):
    m, d = sim.model, sim.data
    mode = target["mode"]
    for t in range(steps):
        if not settle:
            if mode == "body":
                q_ref = ik_step_pos(sim, target["body_id"], pos_des)
            else:
                q_ref = ik_step_midpoint(sim, target["a_id"], target["b_id"], pos_des)
            d.ctrl[:7] = q_ref
        d.ctrl[7] = gripper_cmd
        d.ctrl[:] = clip_ctrl_to_range(m, d.ctrl)
        sim.step(1)
        if (t % log_every) == 0 or t == steps - 1:
            cup = sim.cup_position()
            contact = sim.has_gripper_cup_contact()
            if mode == "body":
                ee = np.array(d.xpos[target["body_id"]])
            else:
                ee = 0.5 * (np.array(d.xpos[target["a_id"]]) + np.array(d.xpos[target["b_id"]]))
            print(
                f"step={len(sim._ctrl_trace):5d} ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                f"cup=({cup[0]:.3f},{cup[1]:.3f},{cup[2]:.3f}) contact={int(contact)}"
            )


def main():
    sim = Sim()
    m, d = sim.model, sim.data

    left_finger_id = m.body("left_finger").id
    right_finger_id = m.body("right_finger").id
    pinch = {"mode": "midpoint", "a_id": left_finger_id, "b_id": right_finger_id}

    cup0 = sim.cup_position()
    print("initial cup", cup0)

    # Open gripper
    pinch0 = 0.5 * (np.array(d.xpos[left_finger_id]) + np.array(d.xpos[right_finger_id]))
    run_phase(sim, pinch, pinch0, steps=200, gripper_cmd=255, settle=True)

    # Pregrasp above cup
    pregrasp = np.array([0.55, 0.15, 0.62])
    run_phase(sim, pinch, pregrasp, steps=1200, gripper_cmd=255)

    # Descend to cup center
    grasp_center = np.array([0.55, 0.15, 0.435])
    run_phase(sim, pinch, grasp_center, steps=1600, gripper_cmd=255)

    # Close gripper while holding pose
    run_phase(sim, pinch, grasp_center, steps=200, gripper_cmd=120)
    run_phase(sim, pinch, grasp_center, steps=500, gripper_cmd=0)

    # Lift and retrieve
    lift = np.array([0.50, 0.15, 0.67])
    run_phase(sim, pinch, lift, steps=1300, gripper_cmd=0)

    pull = np.array([0.38, 0.15, 0.67])
    run_phase(sim, pinch, pull, steps=1700, gripper_cmd=0)

    run_phase(sim, pinch, pull, steps=600, gripper_cmd=0, settle=True)

    cup = sim.cup_position()
    print("final cup", cup, "z>=", SUCCESS_CUP_Z)
    sim.save_final_state("/work/final_state.npz")
    print("saved /work/final_state.npz with ctrl steps", len(sim._ctrl_trace))


if __name__ == "__main__":
    main()
