import numpy as np
import mujoco

from sim import Sim


HAND_BODY = "hand"
LEFT_FINGER_BODY = "left_finger"
RIGHT_FINGER_BODY = "right_finger"
TARGET_DOWN = np.array([0.0, 0.0, -1.0])
SUCCESS_CUP_Z = 0.56
SUCCESS_CUP_X_MAX = 0.42
RETURN_X_CHECKPOINT = 0.48


def solve_ik(model, data, hand_id, left_id, right_id, target_midpoint, seed):
    q = seed.copy()
    qmin = model.actuator_ctrlrange[:7, 0]
    qmax = model.actuator_ctrlrange[:7, 1]

    for _ in range(140):
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


def hold(sim, q, g, steps):
    for _ in range(steps):
        sim.data.ctrl[:7] = q
        sim.data.ctrl[7] = g
        sim.step(1)


def evaluate_trial(sim):
    trace_contact = np.array([entry["cup_contact"] for entry in sim._trace], dtype=float)
    trace_cup = np.array([entry["cup_pos"] for entry in sim._trace], dtype=float).reshape(-1, 3)
    had_grasp = bool(np.any(trace_contact > 0.5))
    best_x = float(np.min(trace_cup[:, 0]))
    best_z = float(np.max(trace_cup[:, 2]))

    settle_contacts = 0
    for _ in range(500):
        sim.step(1)
        settle_contacts += int(sim.has_gripper_cup_contact())
    settle_contact_fraction = settle_contacts / 500.0
    final = sim.cup_position().copy()

    passed = (
        len(sim._ctrl_trace) >= 20
        and had_grasp
        and best_x <= RETURN_X_CHECKPOINT
        and best_z >= SUCCESS_CUP_Z
        and float(final[0]) <= SUCCESS_CUP_X_MAX
        and float(final[2]) >= SUCCESS_CUP_Z
        and settle_contact_fraction >= 0.5
    )

    score = (
        1000.0 * float(passed)
        + 250.0 * float(had_grasp)
        + 220.0 * float(best_x <= RETURN_X_CHECKPOINT)
        + 220.0 * float(best_z >= SUCCESS_CUP_Z)
        + 220.0 * settle_contact_fraction
        + 600.0 * min(float(final[2]), SUCCESS_CUP_Z)
        - 250.0 * max(float(final[0]) - SUCCESS_CUP_X_MAX, 0.0)
    )
    return passed, score, {
        "had_grasp": had_grasp,
        "best_x": best_x,
        "best_z": best_z,
        "final_x": float(final[0]),
        "final_z": float(final[2]),
        "settle_contact_fraction": settle_contact_fraction,
    }


def run_trial(cup, grasp_z, lift_z, return_mid_y, final_y, final_z):
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

    def best_q(target):
        target = np.array(target, dtype=float)
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
    q_pre = best_q([cup[0], cup[1], 0.64])
    q_approach = best_q([cup[0], cup[1], 0.54])
    q_grasp = best_q([cup[0], cup[1], grasp_z])
    q_lift = best_q([cup[0], cup[1], lift_z])
    q_return_mid = best_q([0.46, return_mid_y, lift_z])
    q_return_final = best_q([0.38, final_y, final_z])

    play_segment(sim, q_start, q_pre, 0.0, 255.0, 220)
    play_segment(sim, q_pre, q_approach, 255.0, 255.0, 140)
    play_segment(sim, q_approach, q_grasp, 255.0, 255.0, 160)
    play_segment(sim, q_grasp, q_grasp, 255.0, 35.0, 180)
    play_segment(sim, q_grasp, q_grasp, 35.0, 0.0, 120)
    hold(sim, q_grasp, 0.0, 120)
    play_segment(sim, q_grasp, q_lift, 0.0, 0.0, 240)
    hold(sim, q_lift, 0.0, 120)
    play_segment(sim, q_lift, q_return_mid, 0.0, 0.0, 240)
    hold(sim, q_return_mid, 0.0, 100)
    play_segment(sim, q_return_mid, q_return_final, 0.0, 0.0, 220)
    hold(sim, q_return_final, 0.0, 240)

    passed, score, metrics = evaluate_trial(sim)
    return sim, passed, score, metrics


def main():
    sim0 = Sim()
    cup = sim0.cup_position().copy()

    best_score = None
    best_sim = None
    best_metrics = None
    best_params = None

    for grasp_z in [0.456, 0.462, 0.468]:
        for lift_z in [0.68, 0.72, 0.76]:
            for return_mid_y in [0.12, 0.14]:
                for final_y in [0.10, 0.12, 0.14]:
                    for final_z in [0.66, 0.70, 0.74]:
                        sim, passed, score, metrics = run_trial(cup, grasp_z, lift_z, return_mid_y, final_y, final_z)
                        print(
                            f"trial grasp_z={grasp_z:.3f} lift_z={lift_z:.2f} return_mid_y={return_mid_y:.2f} "
                            f"final_y={final_y:.2f} final_z={final_z:.2f} pass={passed} "
                            f"metrics={metrics}",
                            flush=True,
                        )
                        if best_score is None or passed or score > best_score:
                            best_score = score
                            best_sim = sim
                            best_metrics = metrics
                            best_params = {
                                "grasp_z": grasp_z,
                                "lift_z": lift_z,
                                "return_mid_y": return_mid_y,
                                "final_y": final_y,
                                "final_z": final_z,
                            }
                            best_sim.save_final_state("/work/final_state.npz")
                            print(f"saved best params={best_params} metrics={best_metrics}", flush=True)
                        if passed:
                            print(f"pass params={best_params} metrics={best_metrics}", flush=True)
                            return

    print(f"best params={best_params} metrics={best_metrics}", flush=True)


if __name__ == "__main__":
    main()
