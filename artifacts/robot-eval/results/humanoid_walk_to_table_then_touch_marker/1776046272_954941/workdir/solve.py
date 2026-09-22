import math
import numpy as np
import mujoco

import sim


def run():
    s = sim.Sim()
    model = s.model
    qpos_adr = model.jnt_qposadr
    qvel_adr = model.jnt_dofadr
    act_qpos = [qpos_adr[i + 1] for i in range(model.nu)]
    act_qvel = [qvel_adr[i + 1] for i in range(model.nu)]

    # Tuned posture/gait baseline.
    sign = -1
    hip_amp = 0.15
    knee_amp = 0.10
    lean = 0.16
    left_sp_walk = -1.5
    left_el_walk = 0.0
    right_sp = -0.25
    right_el = 0.9

    q0 = s.home_ctrl().copy()
    for leg in [0, 1]:
        b = 0 if leg == 0 else 5
        q0[b + 2] = -0.25
        q0[b + 3] = 0.8
        q0[b + 4] = -0.55
    q0[10] = lean
    q0[11] = left_sp_walk
    q0[14] = left_el_walk
    q0[15] = right_sp
    q0[18] = right_el

    kp_walk = np.array(
        [120, 120, 180, 240, 120, 120, 120, 180, 240, 120, 120, 60, 50, 40, 40, 60, 50, 40, 40],
        dtype=float,
    )
    kd_walk = np.array(
        [15, 15, 20, 25, 12, 15, 15, 20, 25, 12, 15, 6, 5, 4, 4, 6, 5, 4, 4],
        dtype=float,
    )

    def apply_target(q_des, kp, kd, torque_limit=250.0):
        mujoco.mj_forward(model, s.data)
        q = np.array([s.data.qpos[idx] for idx in act_qpos])
        dq = np.array([s.data.qvel[idx] for idx in act_qvel])
        qacc_des = np.zeros(model.nv)
        qacc_des[np.array(act_qvel)] = kp * (q_des - q) - kd * dq
        s.data.qacc[:] = qacc_des
        mujoco.mj_inverse(model, s.data)
        s.data.ctrl[:] = np.clip(s.data.qfrc_inverse[np.array(act_qvel)], -torque_limit, torque_limit)
        s.step(1)

    # Phase 1: walk to the table while keeping the left arm in a forward-reaching posture.
    for t in range(769):
        phase = sign * 2.0 * math.pi * (t / 120.0)
        sgn = math.sin(phase)
        csgn = math.cos(phase)
        q_des = q0.copy()
        q_des[2] = q0[2] + hip_amp * sgn
        q_des[7] = q0[7] - hip_amp * sgn
        q_des[3] = q0[3] + knee_amp * (1.0 - csgn) / 2.0
        q_des[8] = q0[8] + knee_amp * (1.0 + csgn) / 2.0
        q_des[4] = -q_des[2] - q_des[3]
        q_des[9] = -q_des[7] - q_des[8]
        q_des[1] = 0.04 * sgn
        q_des[6] = -0.04 * sgn
        q_des[10] = lean + 0.06 * sgn
        q_des[11] = left_sp_walk
        q_des[14] = left_el_walk
        q_des[15] = right_sp
        q_des[18] = right_el
        apply_target(q_des, kp_walk, kd_walk)

    # Phase 2: freeze into a stable reaching hold from the successful endpoint.
    q_hold = np.array([s.data.qpos[idx] for idx in act_qpos], dtype=float)
    q_hold[4] = -q_hold[2] - q_hold[3]
    q_hold[9] = -q_hold[7] - q_hold[8]
    q_hold[11] = -1.5
    q_hold[14] = 0.0
    q_hold[15] = right_sp
    q_hold[18] = right_el
    q_hold[10] = q_hold[10]
    kp_hold = np.array(
        [160, 160, 220, 280, 140, 160, 160, 220, 280, 140, 160, 70, 60, 40, 40, 70, 60, 40, 40],
        dtype=float,
    )
    kd_hold = np.array(
        [20, 20, 25, 30, 15, 20, 20, 25, 30, 15, 20, 8, 7, 5, 5, 8, 7, 5, 5],
        dtype=float,
    )
    for _ in range(500):
        apply_target(q_hold, kp_hold, kd_hold)

    print("final pelvis", s.pelvis_position())
    print("final torso_up", s.torso_up())
    print("final touch", s.distance_to_touch())
    print("final target", s.distance_to_target())
    s.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    run()
