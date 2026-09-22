import numpy as np
import mujoco

import sim


def run_attempt(save_path="/work/final_state.npz"):
    s = sim.Sim()
    m = s.model
    d = s.data
    hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    qmin = m.actuator_ctrlrange[:7, 0]
    qmax = m.actuator_ctrlrange[:7, 1]
    nominal_q = np.array([0.0, 0.5, 0.0, -1.0, 0.0, 2.0, 0.4])

    def hold_or_move(hand_target, grip, drawer, steps):
        for _ in range(steps):
            if hand_target is None:
                d.ctrl[:7] = d.qpos[:7]
            else:
                err = hand_target - d.xpos[hand_id]
                jacp = np.zeros((3, m.nv))
                jacr = np.zeros((3, m.nv))
                mujoco.mj_jacBody(m, d, jacp, jacr, hand_id)
                j = jacp[:, :7]
                dq = j.T @ np.linalg.solve(j @ j.T + 3e-3 * np.eye(3), err)
                dq += 0.03 * (nominal_q - d.qpos[:7])
                d.ctrl[:7] = np.clip(d.qpos[:7] + 0.9 * dq, qmin, qmax)
            d.ctrl[7] = grip
            d.ctrl[8] = drawer
            s.step(1)

    hold_or_move(None, grip=255, drawer=1.0, steps=200)
    block = s.block_position().copy()

    for dz in (0.20, 0.14, 0.10):
        hold_or_move(block + np.array([-0.01, 0.0, dz]), grip=255, drawer=1.0, steps=200)

    for grip in (200, 160, 120, 80, 40, 0):
        hold_or_move(block + np.array([-0.01, 0.0, 0.10]), grip=grip, drawer=1.0, steps=120)

    hold_or_move(np.array([0.72, 0.0, 0.62]), grip=0, drawer=1.0, steps=220)
    hold_or_move(np.array([0.68, 0.0, 0.74]), grip=0, drawer=1.0, steps=260)
    hold_or_move(np.array([0.68, 0.0, 0.74]), grip=0, drawer=1.0, steps=300)

    s.save_final_state(save_path)
    return s


def evaluate_replay(ctrl_trace):
    r = sim.Sim()
    max_open = 0.0
    max_z = 0.0
    order_ok = False
    ever_open = False
    for ctrl in ctrl_trace:
        r.data.ctrl[:] = ctrl
        r.step(1)
        max_open = max(max_open, r.drawer_open_amount())
        max_z = max(max_z, r.block_position()[2])
        if r.drawer_open_amount() >= 0.05:
            ever_open = True
        if not order_ok and r.block_position()[2] >= 0.50:
            order_ok = ever_open
    contact_steps = 0
    for _ in range(500):
        r.step(1)
        contact_steps += int(r.has_gripper_block_contact())
    return {
        "steps": len(ctrl_trace),
        "max_open": max_open,
        "max_z": max_z,
        "final_open": r.drawer_open_amount(),
        "final_z": r.block_position()[2],
        "contact_frac": contact_steps / 500.0,
        "order_ok": order_ok,
    }


if __name__ == "__main__":
    sim_out = run_attempt()
    result = evaluate_replay(np.array(sim_out._ctrl_trace))
    print(result)
