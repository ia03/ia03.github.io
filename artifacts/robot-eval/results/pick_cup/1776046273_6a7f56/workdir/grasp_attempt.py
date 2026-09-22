import numpy as np
import mujoco

from sim import Sim


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def solve_hand_position(sim: Sim, target_xyz, q_seed=None, iters=250):
    """Damped least-squares IK on the hand body position."""
    model, data = sim.model, sim.data
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    q = data.qpos.copy() if q_seed is None else q_seed.copy()

    for _ in range(iters):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        pos = data.xpos[hand_id].copy()
        err = np.asarray(target_xyz) - pos
        if np.linalg.norm(err) < 1e-4:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
        J = jacp[:, :7]
        # Damped least squares with a small posture regularizer.
        lam = 1e-2
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
        q[:7] += dq
        for j in range(7):
            lo, hi = model.jnt_range[j]
            q[j] = np.clip(q[j], lo, hi)
    return q


def set_ctrl(sim: Sim, arm_q, gripper_cmd):
    sim.data.ctrl[:7] = np.asarray(arm_q[:7])
    sim.data.ctrl[7] = float(gripper_cmd)


def run_interp(sim: Sim, q_start, q_end, gripper_start, gripper_end, steps):
    for i in range(steps):
        a = smoothstep((i + 1) / steps)
        q = (1.0 - a) * q_start + a * q_end
        g = (1.0 - a) * gripper_start + a * gripper_end
        set_ctrl(sim, q, g)
        sim.step()


def main():
    sim = Sim()

    q_home = sim.data.qpos.copy()
    q_seed = q_home.copy()

    # Solve a three-stage hand trajectory.
    q_pre = solve_hand_position(sim, [0.5, 0.0, 0.55], q_seed=q_seed)
    q_grasp = solve_hand_position(sim, [0.5, 0.0, 0.51], q_seed=q_pre)
    q_lift = solve_hand_position(sim, [0.5, 0.0, 0.75], q_seed=q_grasp)

    # Phase 1: move above the cup with the gripper open.
    run_interp(sim, q_home, q_pre, 255.0, 255.0, 350)

    # Phase 2: descend to the grasp pose while staying open.
    run_interp(sim, q_pre, q_grasp, 255.0, 255.0, 250)

    # Phase 3: close on the cup slowly.
    for i in range(180):
        a = smoothstep((i + 1) / 180.0)
        g = (1.0 - a) * 255.0 + a * 0.0
        set_ctrl(sim, q_grasp, g)
        sim.step()

    # Phase 4: lift and then hold the cup aloft.
    run_interp(sim, q_grasp, q_lift, 0.0, 0.0, 350)
    for _ in range(650):
        set_ctrl(sim, q_lift, 0.0)
        sim.step()

    sim.save_final_state("/work/final_state.npz")

    hand_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cup_pos = sim.cup_position()
    print("final_time", sim.data.time)
    print("final_hand", sim.data.xpos[hand_id])
    print("final_cup", cup_pos)
    print("saved /work/final_state.npz")


if __name__ == "__main__":
    main()
