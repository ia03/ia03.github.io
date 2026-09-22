import numpy as np
import mujoco

import sim


HAND_NAME = "hand"
OPEN_GRIPPER = 255.0
CLOSE_GRIPPER = 0.0


def solve_arm_to_pos(s: sim.Sim, target_pos, q_seed, iters=60, step=0.6, damping=1e-4, reg=1e-3):
    """Solve arm joint targets for a hand-body position using damped least squares."""
    model, data = s.model, s.data
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_NAME)
    q = q_seed.copy()
    q_home = s._initial_qpos.copy()

    for _ in range(iters):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        pos = data.xpos[hand_id].copy()
        err = np.asarray(target_pos, dtype=float) - pos

        Jp = np.zeros((3, model.nv))
        Jr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, Jp, Jr, hand_id)
        A = Jp[:, :7]
        H = A.T @ A + (damping + reg) * np.eye(7)
        g = A.T @ err - reg * (q[:7] - q_home[:7])
        dq = np.linalg.solve(H, g)

        q[:7] += step * dq
        for j in range(7):
            q[j] = np.clip(q[j], model.jnt_range[j, 0], model.jnt_range[j, 1])

        if np.linalg.norm(err) < 1e-4:
            break

    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    return q


def step_hold(s: sim.Sim, q_target, gripper_target, n_steps):
    ctrl = np.zeros(s.model.nu, dtype=float)
    ctrl[:7] = q_target[:7]
    ctrl[7] = gripper_target
    for _ in range(n_steps):
        s.data.ctrl[:] = ctrl
        s.step(1)


def move_to(s: sim.Sim, target_pos, gripper_target, n_steps, q_seed):
    q_target = solve_arm_to_pos(s, target_pos, q_seed)
    step_hold(s, q_target, gripper_target, n_steps)
    return q_target


def main():
    s = sim.Sim()
    q = s.data.qpos.copy()

    # Pregrasp above and slightly to the robot side of the cup.
    waypoints = [
        (np.array([0.60, 0.18, 0.62]), OPEN_GRIPPER, 60),
        (np.array([0.60, 0.18, 0.53]), OPEN_GRIPPER, 80),
        (np.array([0.60, 0.18, 0.53]), CLOSE_GRIPPER, 120),
        (np.array([0.60, 0.18, 0.60]), CLOSE_GRIPPER, 60),
        (np.array([0.55, 0.18, 0.64]), CLOSE_GRIPPER, 60),
        (np.array([0.48, 0.18, 0.66]), CLOSE_GRIPPER, 70),
        (np.array([0.40, 0.18, 0.66]), CLOSE_GRIPPER, 80),
        (np.array([0.35, 0.18, 0.66]), CLOSE_GRIPPER, 140),
        (np.array([0.35, 0.18, 0.66]), CLOSE_GRIPPER, 160),
    ]

    for target_pos, grip, n_steps in waypoints:
        q = solve_arm_to_pos(s, target_pos, q)
        step_hold(s, q, grip, n_steps)
        print(
            f"time={s.data.time:.3f} hand={np.round(s.data.xpos[mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, HAND_NAME)], 3)} "
            f"cup={np.round(s.cup_position(), 3)} contact={s.has_gripper_cup_contact()}"
        )

    # Make sure the final saved state reflects a stable carry posture.
    s.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()
