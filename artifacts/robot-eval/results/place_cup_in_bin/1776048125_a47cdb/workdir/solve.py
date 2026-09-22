import numpy as np
import mujoco
from sim import Sim

BIN_CENTER = np.array([0.70, 0.14, 0.40])


def hand_body_id(sim):
    return mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")


def cup_up_z(sim):
    mat = sim.data.xmat[sim.cup_body_id].reshape(3, 3)
    return float(mat[2, 2])


def rot_err_vec(r_cur, r_des):
    # SO(3) error that drives current basis vectors toward desired basis vectors.
    return 0.5 * (
        np.cross(r_cur[:, 0], r_des[:, 0])
        + np.cross(r_cur[:, 1], r_des[:, 1])
        + np.cross(r_cur[:, 2], r_des[:, 2])
    )


def move_hand(sim, target, steps=200, grip=255, kp=1.2, damping=1e-3, r_des=None, w_rot=0.35):
    m, d = sim.model, sim.data
    hid = hand_body_id(sim)
    target = np.asarray(target, dtype=float)
    for _ in range(steps):
        cur = d.xpos[hid].copy()
        err = target - cur
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, jacr, hid)
        Jp = jacp[:, :7]
        if r_des is None:
            J = Jp
            v = kp * err
        else:
            jacr = jacr[:, :7]
            r_cur = d.xmat[hid].reshape(3, 3).copy()
            er = rot_err_vec(r_cur, r_des)
            J = np.vstack([Jp, w_rot * jacr])
            v = np.concatenate([kp * err, w_rot * 2.0 * er])
        A = J @ J.T + damping * np.eye(J.shape[0])
        dq = J.T @ np.linalg.solve(A, v)
        q_des = d.qpos[:7] + dq
        d.ctrl[:7] = np.clip(q_des, m.actuator_ctrlrange[:7, 0], m.actuator_ctrlrange[:7, 1])
        d.ctrl[7] = grip
        sim.step(1)


def hold(sim, steps=100, grip=0):
    for _ in range(steps):
        sim.data.ctrl[7] = grip
        sim.step(1)


def progress(sim):
    cup = sim.cup_position()
    inside = abs(cup[0] - 0.70) <= 0.046 and abs(cup[1] - 0.14) <= 0.046 and cup[2] >= 0.42
    tr = sim._trace
    contact_hist = np.array([t["cup_contact"] for t in tr], dtype=float)
    best_z = max(float(t["cup_pos"][2]) for t in tr)
    c_last10 = contact_hist[-10:].mean() if len(contact_hist) >= 10 else contact_hist.mean()
    dist0 = np.linalg.norm(np.array([0.48, -0.12]) - np.array([0.70, 0.14]))
    distf = np.linalg.norm(cup[:2] - np.array([0.70, 0.14]))
    grasp = 1.0 if contact_hist.max() > 0 else 0.0
    lift = np.clip((best_z - 0.435) / (0.50 - 0.435), 0, 1)
    inside_p = 1.0 if inside else 0.0
    placement = 1.0 if inside else np.clip(1 - distf / dist0, 0, 1)
    release = np.clip(1 - c_last10 / 0.1, 0, 1)
    upright = np.clip(abs(cup_up_z(sim)) / 0.85, 0, 1)
    score = 0.05*grasp + 0.10*lift + 0.40*inside_p + 0.10*placement + 0.10*release + 0.25*upright
    return {
        "cup": cup,
        "inside": inside,
        "best_z": best_z,
        "contact_last10": c_last10,
        "up_z": cup_up_z(sim),
        "score": float(score),
    }


def run_once(params):
    sim = Sim()
    hid = hand_body_id(sim)
    r_des = sim.data.xmat[hid].reshape(3, 3).copy()

    move_hand(sim, [0.48, -0.12, 0.62], steps=220, grip=255, r_des=r_des)
    move_hand(sim, [0.48, -0.12, params["grasp_h"]], steps=220, grip=255, r_des=r_des)

    hold(sim, steps=params["squeeze_steps"], grip=0)
    move_hand(sim, [0.55, 0.00, 0.63], steps=220, grip=0, r_des=r_des)
    move_hand(sim, [0.70, 0.14, 0.62], steps=240, grip=0, r_des=r_des)
    move_hand(sim, [0.70, 0.14, params["drop_h"]], steps=180, grip=0, r_des=r_des)

    hold(sim, steps=params["open_wait1"], grip=255)
    move_hand(sim, [0.70, 0.14, 0.68], steps=220, grip=255, r_des=r_des)
    hold(sim, steps=params["settle"], grip=255)

    met = progress(sim)
    return sim, met


if __name__ == "__main__":
    params = {
        "grasp_h": 0.495,
        "drop_h": 0.53,
        "squeeze_steps": 180,
        "open_wait1": 120,
        "settle": 520,
    }
    sim, met = run_once(params)
    sim.save_final_state('/work/final_state.npz')
    print('metrics', met)
    print('saved', '/work/final_state.npz')
