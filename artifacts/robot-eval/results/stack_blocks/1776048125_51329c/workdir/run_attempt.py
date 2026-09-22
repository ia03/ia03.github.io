import numpy as np
import mujoco
from sim import Sim

sim = Sim()
m, d = sim.model, sim.data
hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
lf_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
rf_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

qmin = m.jnt_range[:7, 0]
qmax = m.jnt_range[:7, 1]

def finger_mid():
    return 0.5 * (d.xpos[lf_id] + d.xpos[rf_id])

def set_ctrl(q=None, grip=255.0):
    if q is None:
        q = d.qpos[:7]
    d.ctrl[:7] = q
    d.ctrl[7] = grip

def hold(steps, grip):
    for _ in range(steps):
        set_ctrl(d.qpos[:7], grip)
        sim.step(1)

def move_hand_to(target, steps=800, grip=255.0, kp=4.0, damp=1e-3, tol=0.0035):
    target = np.array(target, dtype=float)
    for _ in range(steps):
        mujoco.mj_forward(m, d)
        p = d.xpos[hand_id].copy()
        err = target - p
        if np.linalg.norm(err) < tol:
            set_ctrl(d.qpos[:7], grip)
            sim.step(1)
            continue
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, jacr, hand_id)
        J = jacp[:, :7]
        A = J @ J.T + damp * np.eye(3)
        dq = J.T @ np.linalg.solve(A, err * kp)
        qdes = np.clip(d.qpos[:7] + np.clip(dq, -0.05, 0.05), qmin, qmax)
        set_ctrl(qdes, grip)
        sim.step(1)

def settle(n=500):
    hold(n, d.ctrl[7])

def run_once():
    sim.reset()
    hold(120, 255)  # open and stabilize

    red = sim.block_positions()["red"].copy()
    green = sim.block_positions()["green"].copy()

    # Approach red block from above.
    move_hand_to([red[0], red[1], 0.66], steps=1200, grip=255)
    move_hand_to([red[0], red[1], 0.57], steps=900, grip=255)

    # Descend for grasp.
    move_hand_to([red[0], red[1], 0.505], steps=1000, grip=255)
    hold(260, 0)  # close
    hold(120, 0)

    # Lift and translate to green.
    move_hand_to([red[0], red[1], 0.66], steps=1100, grip=0)
    green = sim.block_positions()["green"].copy()
    move_hand_to([green[0], green[1], 0.67], steps=1600, grip=0)

    # Place onto green.
    move_hand_to([green[0], green[1], 0.545], steps=1100, grip=0)
    hold(120, 0)
    hold(260, 255)  # release

    # Retreat and settle.
    move_hand_to([green[0], green[1], 0.69], steps=900, grip=255)
    settle(700)


def metrics():
    pos = sim.block_positions()
    dxyz = pos["red"] - pos["green"]
    return dxyz, pos

if __name__ == "__main__":
    run_once()
    dxyz, pos = metrics()
    print("final red", pos["red"])
    print("final green", pos["green"])
    print("dxyz", dxyz)
    sim.save_final_state("/work/final_state.npz")
    print("saved /work/final_state.npz, t=", d.time, "steps=", len(sim._ctrl_trace))
