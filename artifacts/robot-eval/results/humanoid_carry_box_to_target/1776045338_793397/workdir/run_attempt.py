import math
import numpy as np

from sim import Sim, TARGET_XY, MIN_PELVIS_Z, MIN_TORSO_UP, MIN_BOX_Z, MAX_BOX_TO_PELVIS


def metrics(sim, settle_steps=500):
    pelvis_checkpoints = [0.05, 0.09, 0.13]
    box_checkpoints = [0.00, 0.04, 0.08]
    trace = sim._trace
    pelvis_x = np.array([t["pelvis_pos"][0] for t in trace])
    box_x = np.array([t["box_pos"][0] for t in trace])
    stable = np.array([
        (
            t["pelvis_pos"][2] >= MIN_PELVIS_Z
            and t["torso_up"] >= MIN_TORSO_UP
            and t["box_pos"][2] >= MIN_BOX_Z
            and t["box_distance_to_pelvis"] <= MAX_BOX_TO_PELVIS
        )
        for t in trace
    ], dtype=float)
    replay_ok = {
        "replay_steps": len(sim._ctrl_trace),
        "pelvis_hits": [bool(np.any(pelvis_x >= c)) for c in pelvis_checkpoints],
        "box_hits": [bool(np.any(box_x >= c)) for c in box_checkpoints],
        "stable_frac": float(stable.mean()) if len(stable) else 0.0,
    }

    qpos = sim.data.qpos.copy()
    qvel = sim.data.qvel.copy()
    ctrl = sim.data.ctrl.copy()
    settle = []
    for _ in range(settle_steps):
        sim.step(1)
        settle.append(
            (
                sim.pelvis_position().copy(),
                sim.torso_up(),
                sim.box_position().copy(),
                sim.box_distance_to_pelvis(),
            )
        )
    settle_pelvis = np.array([s[0] for s in settle])
    settle_torso_up = np.array([s[1] for s in settle])
    settle_box = np.array([s[2] for s in settle])
    settle_box_d = np.array([s[3] for s in settle])
    replay_ok.update(
        {
            "final_pelvis_dist": float(np.linalg.norm(settle_pelvis[-1, :2] - TARGET_XY)),
            "final_box_dist": float(np.linalg.norm(settle_box[-1, :2] - TARGET_XY)),
            "min_settle_pelvis_z": float(settle_pelvis[:, 2].min()),
            "min_settle_torso_up": float(settle_torso_up.min()),
            "min_settle_box_z": float(settle_box[:, 2].min()),
            "max_settle_box_d": float(settle_box_d.max()),
        }
    )
    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = qvel
    sim.data.ctrl[:] = ctrl
    return replay_ok


def pose(home, *, lsp=0.0, lsr=0.0, lsy=0.0, le=0.0, rsp=0.0, rsr=0.0, rsy=0.0, re=0.0,
         torso=0.0, lhp=None, lkn=None, lak=None, rhp=None, rkn=None, rak=None,
         lhr=0.0, rhr=0.0, lhy=0.0, rhy=0.0):
    q = home.copy()
    q[0] = lhy
    q[1] = lhr
    q[2] = home[2] if lhp is None else lhp
    q[3] = home[3] if lkn is None else lkn
    q[4] = home[4] if lak is None else lak
    q[5] = rhy
    q[6] = rhr
    q[7] = home[7] if rhp is None else rhp
    q[8] = home[8] if rkn is None else rkn
    q[9] = home[9] if rak is None else rak
    q[10] = torso
    q[11:15] = [lsp, lsr, lsy, le]
    q[15:19] = [rsp, rsr, rsy, re]
    return q


def run_policy():
    sim = Sim()
    home = sim.home_ctrl()

    # Mandatory early save with a stable placeholder trajectory.
    sim.data.ctrl[:] = home
    sim.step(40)
    sim.save_final_state("/work/final_state.npz")

    phases = []
    # Fold arms inward around the box while staying close to home posture.
    phases.append((120, pose(home, lsp=0.9, lsr=0.45, le=1.2, rsp=0.9, rsr=-0.45, re=1.2)))
    phases.append((100, pose(home, lsp=1.15, lsr=0.55, lsy=-0.2, le=1.45,
                                  rsp=1.15, rsr=-0.55, rsy=0.2, re=1.45, torso=0.08)))
    # Small forward-leaning shuffle.
    phases.append((80, pose(home, lsp=1.15, lsr=0.55, lsy=-0.2, le=1.45,
                                 rsp=1.15, rsr=-0.55, rsy=0.2, re=1.45,
                                 torso=0.15, lhp=-0.55, lkn=0.95, lak=-0.4,
                                 rhp=-0.35, rkn=0.75, rak=-0.35, lhr=0.03, rhr=-0.03)))
    phases.append((80, pose(home, lsp=1.15, lsr=0.55, lsy=-0.2, le=1.45,
                                 rsp=1.15, rsr=-0.55, rsy=0.2, re=1.45,
                                 torso=0.15, lhp=-0.35, lkn=0.75, lak=-0.35,
                                 rhp=-0.55, rkn=0.95, rak=-0.4, lhr=-0.03, rhr=0.03)))
    phases.append((120, pose(home, lsp=1.1, lsr=0.55, le=1.35, rsp=1.1, rsr=-0.55, re=1.35,
                                  torso=0.10, lhp=-0.45, lkn=0.85, lak=-0.4,
                                  rhp=-0.45, rkn=0.85, rak=-0.4)))
    phases.append((200, pose(home, lsp=1.0, lsr=0.5, le=1.25, rsp=1.0, rsr=-0.5, re=1.25,
                                  torso=0.05)))

    for steps, target in phases:
        for i in range(steps):
            sim.data.ctrl[:] = target
            sim.step(1)

    info = metrics(sim)
    print(info)
    sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    run_policy()
