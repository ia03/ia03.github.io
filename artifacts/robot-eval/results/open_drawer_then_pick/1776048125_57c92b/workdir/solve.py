import numpy as np
import mujoco
from sim import Sim

OPEN_THRESH = 0.05
HEIGHT_THRESH = 0.595


def orientation_error(R, Rd):
    return 0.5 * (
        np.cross(R[:, 0], Rd[:, 0])
        + np.cross(R[:, 1], Rd[:, 1])
        + np.cross(R[:, 2], Rd[:, 2])
    )


def run_trial(cfg):
    sim = Sim()
    m, d = sim.model, sim.data
    hid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")

    # Seed to near top-down-ish pose.
    seed_q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    for _ in range(250):
        d.ctrl[:7] = seed_q
        d.ctrl[7] = 255.0
        d.ctrl[8] = 0.0
        sim.step(1)

    R_target = d.xmat[hid].reshape(3, 3).copy()
    qcmd = d.qpos[:7].copy()
    dof = [m.jnt_dofadr[i] for i in range(7)]

    def control_to(hand_target, grip, steps, kp=3.0):
        nonlocal qcmd
        for _ in range(steps):
            jacp = np.zeros((3, m.nv))
            jacr = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jacp, jacr, hid)
            Jp = jacp[:, dof]
            Jr = jacr[:, dof]
            R = d.xmat[hid].reshape(3, 3)
            ep = hand_target - d.xpos[hid]
            eo = orientation_error(R, R_target)
            J = np.vstack([Jp, cfg["ori_w"] * Jr])
            v = np.hstack([kp * ep, cfg["ori_w"] * 1.8 * eo])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), v)
            n = np.linalg.norm(dq)
            if n > cfg["max_dq"]:
                dq *= cfg["max_dq"] / n
            qcmd = qcmd + dq
            for i in range(7):
                lo, hi = m.actuator_ctrlrange[i]
                qcmd[i] = np.clip(qcmd[i], lo, hi)

            drawer_err = cfg["drawer_target"] - sim.drawer_open_amount()
            d.ctrl[8] = np.clip(20.0 * drawer_err, -1.0, 1.0)
            d.ctrl[:7] = qcmd
            d.ctrl[7] = grip
            sim.step(1)

    # Phase 1: pre-shape and open drawer above threshold.
    control_to(np.array([0.52, 0.0, 0.60]), 255.0, 180, kp=2.6)

    # Phase 2: above block.
    for _ in range(180):
        b = sim.block_position()
        tgt = b + np.array([cfg["pre_x"], cfg["pre_y"], cfg["pre_z"]])
        control_to(tgt, 255.0, 1, kp=3.0)

    # Phase 3: descend.
    for _ in range(220):
        b = sim.block_position()
        tgt = b + np.array([cfg["grasp_x"], cfg["grasp_y"], cfg["grasp_z"]])
        control_to(tgt, cfg["grip_open"], 1, kp=2.0)

    # Phase 4: close.
    for t in range(200):
        b = sim.block_position()
        tgt = b + np.array([cfg["grasp_x"], cfg["grasp_y"], cfg["grasp_z"]])
        g = max(cfg["grip_close"], cfg["grip_open"] - t * cfg["close_rate"])
        control_to(tgt, g, 1, kp=1.6)

    # Phase 5: lift and retreat.
    for _ in range(350):
        b = sim.block_position()
        tgt = np.array([cfg["lift_x"], cfg["lift_y"], max(cfg["lift_z"], b[2] + 0.18)])
        control_to(tgt, cfg["grip_close"], 1, kp=2.2)

    # Extra hold time so replay end-state is meaningful.
    control_to(np.array([cfg["lift_x"], cfg["lift_y"], cfg["lift_z"]]), cfg["grip_close"], 180, kp=1.3)

    # Gather replay metrics from trace.
    tr_draw = np.array([e["drawer_open"] for e in sim._trace])
    tr_bz = np.array([e["block_pos"][2] for e in sim._trace])
    max_draw = float(np.max(tr_draw))
    max_bz = float(np.max(tr_bz))

    # Order check over sampled trace.
    open_idx = np.where(tr_draw >= OPEN_THRESH)[0]
    lift_idx = np.where(tr_bz >= 0.50)[0]
    order_ok = bool(len(open_idx) and len(lift_idx) and open_idx[0] < lift_idx[0])

    # Settle.
    contact_count = 0
    for _ in range(500):
        sim.step(1)
        if sim.has_gripper_block_contact():
            contact_count += 1
    final_draw = sim.drawer_open_amount()
    final_bz = float(sim.block_position()[2])
    contact_frac = contact_count / 500.0

    drawer_progress = np.clip(max_draw / OPEN_THRESH, 0, 1)
    retrieval_progress = np.clip((max_bz - 0.435) / (0.50 - 0.435), 0, 1)
    ordered_retrieval = retrieval_progress if order_ok else 0.0
    final_height_progress = np.clip((final_bz - 0.43) / (HEIGHT_THRESH - 0.43), 0, 1)
    hold_progress = np.clip(contact_frac / 0.4, 0, 1)
    final_drawer_progress = np.clip(final_draw / OPEN_THRESH, 0, 1)
    progress = (
        0.22 * drawer_progress
        + 0.28 * ordered_retrieval
        + 0.22 * final_height_progress
        + 0.18 * hold_progress
        + 0.10 * final_drawer_progress
    )

    pass_like = (
        len(sim._ctrl_trace) >= 20
        and order_ok
        and max_draw >= OPEN_THRESH
        and final_draw >= OPEN_THRESH
        and final_bz >= HEIGHT_THRESH
        and contact_frac >= 0.4
    )

    return {
        "sim": sim,
        "progress": float(progress),
        "pass_like": bool(pass_like),
        "max_draw": max_draw,
        "max_bz": max_bz,
        "final_draw": final_draw,
        "final_bz": final_bz,
        "contact_frac": contact_frac,
        "order_ok": order_ok,
    }


def main():
    configs = [
        dict(ori_w=0.20, max_dq=0.03, drawer_target=0.060, pre_x=0.00, pre_y=0.00, pre_z=0.16, grasp_x=0.00, grasp_y=0.00, grasp_z=0.075, grip_open=170.0, grip_close=8.0, close_rate=0.9, lift_x=0.56, lift_y=0.00, lift_z=0.78),
        dict(ori_w=0.12, max_dq=0.03, drawer_target=0.055, pre_x=-0.01, pre_y=0.00, pre_z=0.17, grasp_x=-0.005, grasp_y=0.00, grasp_z=0.070, grip_open=180.0, grip_close=5.0, close_rate=0.95, lift_x=0.55, lift_y=0.00, lift_z=0.80),
        dict(ori_w=0.08, max_dq=0.035, drawer_target=0.052, pre_x=-0.015, pre_y=0.00, pre_z=0.16, grasp_x=-0.01, grasp_y=0.00, grasp_z=0.065, grip_open=190.0, grip_close=2.0, close_rate=1.1, lift_x=0.54, lift_y=0.00, lift_z=0.82),
        dict(ori_w=0.18, max_dq=0.028, drawer_target=0.065, pre_x=0.00, pre_y=0.00, pre_z=0.15, grasp_x=0.00, grasp_y=0.00, grasp_z=0.080, grip_open=160.0, grip_close=10.0, close_rate=0.8, lift_x=0.58, lift_y=0.00, lift_z=0.76),
    ]

    best = None
    for i, cfg in enumerate(configs):
        out = run_trial(cfg)
        print(f"trial {i}: progress={out['progress']:.3f} pass_like={out['pass_like']} max_draw={out['max_draw']:.3f} max_bz={out['max_bz']:.3f} final_bz={out['final_bz']:.3f} contact={out['contact_frac']:.3f} order={out['order_ok']}")
        if best is None or out["progress"] > best["progress"]:
            best = out
            out["sim"].save_final_state('/work/final_state.npz')
            print('  saved new best /work/final_state.npz')
        if out["pass_like"]:
            break

    print('best progress', best['progress'])


if __name__ == '__main__':
    main()
