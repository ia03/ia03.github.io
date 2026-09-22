import numpy as np
import mujoco
from sim import Sim


def hand_pos(sim):
    hid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    return sim.data.xpos[hid].copy()


def jac_hand_pos(sim):
    m, d = sim.model, sim.data
    hid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jacBodyCom(m, d, jacp, None, hid)
    dofs = [m.jnt_dofadr[i] for i in range(7)]
    return jacp[:, dofs]


def ik_step(sim, target, qcmd, gain=4.0, damp=1e-3, max_step=0.03):
    d, m = sim.data, sim.model
    cur = hand_pos(sim)
    err = target - cur
    J = jac_hand_pos(sim)
    A = J @ J.T + damp * np.eye(3)
    v = gain * err
    dq = J.T @ np.linalg.solve(A, v)
    n = np.linalg.norm(dq)
    if n > max_step:
        dq *= max_step / n
    qcmd = qcmd + dq
    for i in range(7):
        lo, hi = m.actuator_ctrlrange[i]
        qcmd[i] = np.clip(qcmd[i], lo, hi)
    return qcmd, np.linalg.norm(err)


def run():
    sim = Sim()
    d = sim.data
    qcmd = d.qpos[:7].copy()

    # Stage 1: open drawer and move near front.
    for t in range(260):
        target = np.array([0.60, -0.01, 0.62])
        qcmd, _ = ik_step(sim, target, qcmd, gain=3.0)
        d.ctrl[:7] = qcmd
        d.ctrl[7] = 255.0
        d.ctrl[8] = 1.0
        sim.step(1)

    # Stage 2: align above block.
    for t in range(280):
        b = sim.block_position()
        target = b + np.array([0.0, 0.0, 0.16])
        qcmd, _ = ik_step(sim, target, qcmd, gain=3.2)
        d.ctrl[:7] = qcmd
        d.ctrl[7] = 255.0
        d.ctrl[8] = 1.0
        sim.step(1)

    # Stage 3: descend for grasp.
    for t in range(240):
        b = sim.block_position()
        target = b + np.array([0.0, 0.0, 0.09])
        qcmd, _ = ik_step(sim, target, qcmd, gain=2.5)
        d.ctrl[:7] = qcmd
        d.ctrl[7] = 210.0
        d.ctrl[8] = 1.0
        sim.step(1)

    # Stage 4: close fingers.
    for t in range(260):
        b = sim.block_position()
        target = b + np.array([0.0, 0.0, 0.09])
        qcmd, _ = ik_step(sim, target, qcmd, gain=2.0)
        d.ctrl[:7] = qcmd
        d.ctrl[7] = max(0.0, 210.0 - 1.0 * t)
        d.ctrl[8] = 1.0
        sim.step(1)

    # Stage 5: lift while holding drawer open.
    for t in range(420):
        b = sim.block_position()
        target = np.array([b[0], b[1], 0.78])
        qcmd, _ = ik_step(sim, target, qcmd, gain=2.2)
        d.ctrl[:7] = qcmd
        d.ctrl[7] = 6.0
        d.ctrl[8] = 1.0
        sim.step(1)

    # Brief settle under control.
    for _ in range(180):
        d.ctrl[:7] = qcmd
        d.ctrl[7] = 6.0
        d.ctrl[8] = 1.0
        d.ctrl[8] = 1.0
        sim.step(1)

    sim.save_final_state('/work/final_state.npz')
    print('saved /work/final_state.npz')
    print('drawer_open', sim.drawer_open_amount())
    print('block', sim.block_position())
    print('contact', sim.has_gripper_block_contact())
    print('time', sim.data.time)
    print('ctrl_steps', len(sim._ctrl_trace))


if __name__ == '__main__':
    run()
