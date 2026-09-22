import numpy as np
import mujoco

from sim import Sim


def solve_ik_to_position(sim, target, q_init, iters=25, damping=1e-2):
    q = q_init.copy()
    hand_id = sim.model.body("hand").id
    for _ in range(iters):
        sim.data.qpos[:7] = q
        mujoco.mj_forward(sim.model, sim.data)
        pos = sim.data.xpos[hand_id].copy()
        err = target - pos
        if np.linalg.norm(err) < 1e-4:
            break
        Jp = np.zeros((3, sim.model.nv))
        Jr = np.zeros((3, sim.model.nv))
        mujoco.mj_jacBody(sim.model, sim.data, Jp, Jr, hand_id)
        J = Jp[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
        q += dq
        for i in range(7):
            lo, hi = sim.model.jnt_range[i]
            q[i] = np.clip(q[i], lo + 1e-3, hi - 1e-3)
    return q


def move_with_ik(sim, target, steps, finger_ctrl, drawer_ctrl, q_seed=None, blend=0.5):
    if q_seed is None:
        q_seed = sim.data.qpos[:7].copy()
    q_target = q_seed.copy()
    for _ in range(steps):
        q_new = solve_ik_to_position(sim, target, q_target, iters=10)
        q_target = blend * q_new + (1.0 - blend) * q_target
        sim.data.ctrl[:7] = q_target
        sim.data.ctrl[7] = finger_ctrl
        sim.data.ctrl[8] = drawer_ctrl
        sim.step()
    return q_target


def open_drawer(sim, steps=120):
    for _ in range(steps):
        sim.data.ctrl[:7] = sim.data.qpos[:7]
        sim.data.ctrl[7] = 0.0
        sim.data.ctrl[8] = 1.0
        sim.step()


def main():
    sim = Sim()
    hand_id = sim.model.body("hand").id

    # Stage 1: open drawer fully while keeping the arm out of the way.
    open_drawer(sim, steps=130)

    # Stage 2: move to a hover pose above the exposed block.
    q = sim.data.qpos[:7].copy()
    # The gripper opens at the high control value on this model.
    q = move_with_ik(sim, np.array([0.76, -0.02, 0.60]), 90, finger_ctrl=255.0, drawer_ctrl=1.0, q_seed=q)

    # Stage 3: descend to grasp height, still open.
    q = move_with_ik(sim, np.array([0.76, -0.02, 0.40]), 110, finger_ctrl=255.0, drawer_ctrl=1.0, q_seed=q)

    # Stage 4: close the gripper around the block.
    for _ in range(140):
        sim.data.ctrl[:7] = q
        sim.data.ctrl[7] = 0.0
        sim.data.ctrl[8] = 1.0
        sim.step()

    # Stage 5: lift in small increments so the block stays captured.
    for z, steps in [(0.46, 60), (0.52, 60), (0.58, 70), (0.64, 80), (0.70, 90)]:
        q = move_with_ik(sim, np.array([0.76, -0.02, z]), steps, finger_ctrl=0.0, drawer_ctrl=1.0, q_seed=q)
    for _ in range(140):
        sim.data.ctrl[:7] = q
        sim.data.ctrl[7] = 0.0
        sim.data.ctrl[8] = 1.0
        sim.step()

    print("time", sim.data.time)
    print("drawer", sim.drawer_open_amount())
    print("block", sim.block_position())
    print("hand", sim.data.xpos[hand_id])
    print("fingers", sim.data.qpos[7:9])
    print("contact", sim.has_gripper_block_contact())
    sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()
