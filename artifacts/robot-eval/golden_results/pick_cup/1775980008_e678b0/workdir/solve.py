import numpy as np
import mujoco

from sim import Sim


HAND_BODY = "hand"
LEFT_FINGER_BODY = "left_finger"
RIGHT_FINGER_BODY = "right_finger"
TARGET_DOWN = np.array([0.0, 0.0, -1.0])


def solve_ik(model, data, hand_id, left_id, right_id, target_midpoint, seed):
    q = seed.copy()
    qmin = model.actuator_ctrlrange[:7, 0]
    qmax = model.actuator_ctrlrange[:7, 1]

    for _ in range(120):
        data.qpos[:7] = q
        data.qpos[7:9] = 0.04
        mujoco.mj_forward(model, data)

        hand_z = data.xmat[hand_id].reshape(3, 3)[:, 2]
        midpoint = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
        pos_err = target_midpoint - midpoint
        ori_err = np.cross(hand_z, TARGET_DOWN)

        jacp_l = np.zeros((3, model.nv))
        jacr_l = np.zeros((3, model.nv))
        jacp_r = np.zeros((3, model.nv))
        jacr_r = np.zeros((3, model.nv))
        jacp_h = np.zeros((3, model.nv))
        jacr_h = np.zeros((3, model.nv))

        mujoco.mj_jacBodyCom(model, data, jacp_l, jacr_l, left_id)
        mujoco.mj_jacBodyCom(model, data, jacp_r, jacr_r, right_id)
        mujoco.mj_jacBody(model, data, jacp_h, jacr_h, hand_id)

        jac = np.vstack([0.5 * (jacp_l[:, :7] + jacp_r[:, :7]), 0.35 * jacr_h[:, :7]])
        err = np.concatenate([pos_err, 0.35 * ori_err])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), err)
        q = np.clip(q + 0.8 * dq, qmin, qmax)

    return q


def play_segment(sim, q0, q1, g0, g1, steps):
    for i in range(steps):
        u = (i + 1) / steps
        sim.data.ctrl[:7] = (1.0 - u) * q0 + u * q1
        sim.data.ctrl[7] = (1.0 - u) * g0 + u * g1
        sim.step(1)


def main():
    sim = Sim()
    model = sim.model
    data = sim.data

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_FINGER_BODY)
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_FINGER_BODY)

    seeds = [
        np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.0]),
        np.array([0.08, -0.19, -0.07, -1.88, -0.01, 1.69, 0.5]),
        np.array([-0.08, -0.19, 0.07, -1.88, 0.01, 1.69, -0.5]),
    ]

    def best_q(z):
        target = np.array([0.5, 0.0, z])
        candidates = []
        for seed in seeds:
            q = solve_ik(model, data, hand_id, left_id, right_id, target, seed)
            data.qpos[:7] = q
            data.qpos[7:9] = 0.04
            mujoco.mj_forward(model, data)
            midpoint = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
            hand_z = data.xmat[hand_id].reshape(3, 3)[:, 2]
            err = np.linalg.norm(midpoint - target) + 0.2 * np.linalg.norm(np.cross(hand_z, TARGET_DOWN))
            candidates.append((err, q))
        return min(candidates, key=lambda item: item[0])[1]

    q_start = np.zeros(7)
    q_above = best_q(0.54)
    q_grasp = best_q(0.468)
    q_lift = best_q(0.60)

    play_segment(sim, q_start, q_above, 0.0, 255.0, 220)
    play_segment(sim, q_above, q_grasp, 255.0, 255.0, 120)
    play_segment(sim, q_grasp, q_grasp, 255.0, 40.0, 160)
    play_segment(sim, q_grasp, q_grasp, 40.0, 0.0, 80)
    play_segment(sim, q_grasp, q_lift, 0.0, 0.0, 180)
    play_segment(sim, q_lift, q_lift, 0.0, 0.0, 250)

    for _ in range(150):
        sim.data.ctrl[:7] = q_lift
        sim.data.ctrl[7] = 0.0
        sim.step(1)

    sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()
