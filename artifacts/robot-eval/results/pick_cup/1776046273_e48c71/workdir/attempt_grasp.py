import numpy as np
import mujoco

from sim import Sim, SUCCESS_CUP_Z


ARM_DOF = 7
GRIPPER_IDX = 7
HAND_BODY = "hand"
CUP_GEOM = "cup_geom"
FINGER_GEOM_IDS = list(range(66, 82))
CUP_GEOM_ID = 83


def rotmat_to_axis_angle(R):
    tr = np.trace(R)
    cosang = (tr - 1.0) * 0.5
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.arccos(cosang)
    if ang < 1e-8:
        return np.zeros(3)
    denom = 2.0 * np.sin(ang)
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    ) / denom
    return axis * ang


def hand_body_id(sim):
    return mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)


def solve_arm_position(sim, target_pos, q_seed, max_iter=180, step=0.5, lam=1e-2):
    q = q_seed.copy()
    bid = hand_body_id(sim)
    q_ref = q_seed.copy()
    for _ in range(max_iter):
        sim.data.qpos[:ARM_DOF] = q
        sim.data.qvel[:] = 0
        mujoco.mj_forward(sim.model, sim.data)
        err = target_pos - sim.data.xpos[bid]
        if np.linalg.norm(err) < 1e-4:
            break
        jacp = np.zeros((3, sim.model.nv))
        jacr = np.zeros((3, sim.model.nv))
        mujoco.mj_jacBody(sim.model, sim.data, jacp, jacr, bid)
        j = jacp[:, :ARM_DOF]
        a = j.T @ j + lam * np.eye(ARM_DOF)
        b = j.T @ err + 0.01 * (q_ref - q)
        dq = np.linalg.solve(a, b)
        q = np.clip(
            q + step * dq,
            sim.model.jnt_range[:ARM_DOF, 0],
            sim.model.jnt_range[:ARM_DOF, 1],
        )
    return q


def set_ctrl(sim, arm_q, grip):
    sim.data.ctrl[:ARM_DOF] = arm_q
    sim.data.ctrl[GRIPPER_IDX] = grip


def step_for(sim, n, arm_q, grip):
    for _ in range(n):
        set_ctrl(sim, arm_q, grip)
        sim.step(1)


def interp_run(sim, q_start, q_end, n, grip):
    for i in range(n):
        t = (i + 1) / n
        q = (1.0 - t) * q_start + t * q_end
        set_ctrl(sim, q, grip)
        sim.step(1)


def contact_fraction(sim):
    count = 0
    total = 0
    for _ in range(500):
        set_ctrl(sim, sim.data.ctrl[:ARM_DOF].copy(), sim.data.ctrl[GRIPPER_IDX])
        sim.step(1)
        total += 1
        hit = False
        for ci in range(sim.data.ncon):
            c = sim.data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            if (g1 == CUP_GEOM_ID and g2 in FINGER_GEOM_IDS) or (
                g2 == CUP_GEOM_ID and g1 in FINGER_GEOM_IDS
            ):
                hit = True
                break
        if hit:
            count += 1
    return count / total


def main():
    sim = Sim()

    # Phase 1: open the gripper a bit while keeping the arm still.
    q_arm = sim.data.qpos[:ARM_DOF].copy()
    step_for(sim, 60, q_arm, 255)

    # Plan waypoints from the live state.
    q_start = sim.data.qpos[:ARM_DOF].copy()
    q_pre = solve_arm_position(sim, np.array([0.50, 0.00, 0.62]), q_start)
    q_grasp = solve_arm_position(sim, np.array([0.50, 0.00, 0.54]), q_pre)
    q_lift = solve_arm_position(sim, np.array([0.50, 0.00, 0.70]), q_grasp)

    # Execute a conservative grasp sequence.
    interp_run(sim, q_start, q_pre, 160, 255)
    interp_run(sim, q_pre, q_grasp, 120, 255)
    step_for(sim, 30, q_grasp, 255)
    interp_run(sim, q_grasp, q_grasp, 100, 0)
    interp_run(sim, q_grasp, q_lift, 220, 0)
    step_for(sim, 160, q_lift, 0)

    # Evaluate the replayed trajectory with an additional settle window.
    final_z = sim.cup_position()[2]
    settle_contact = contact_fraction(sim)
    print("replay_end_cup_z", final_z)
    print("settle_contact_fraction", settle_contact)
    print("success_height_threshold", SUCCESS_CUP_Z)

    sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()
