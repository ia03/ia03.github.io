import numpy as np, mujoco, sim

def damped_pinv(J, lam=0.05):
    JJ = J @ J.T
    A = JJ + (lam ** 2) * np.eye(JJ.shape[0])
    return J.T @ np.linalg.solve(A, np.eye(JJ.shape[0]))

def run_once():
    s = sim.Sim()
    hand = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    finger_target = 0.02

    waypoints = [
        np.array([0.45, -0.10, 0.90]),
        np.array([0.45, -0.10, 0.58]),
        np.array([0.47, -0.10, 0.54]),
        np.array([0.56, -0.10, 0.52]),
        np.array([0.58, -0.10, 0.52]),
        np.array([0.60, -0.10, 0.52]),
    ]
    dof_idx = np.array([0,1,2,3,4,5,6])

    def move_to(pos_target, gain=0.5, steps=220):
        for _ in range(steps):
            x = s.data.xpos[hand]
            err = pos_target - x
            if np.linalg.norm(err) < 1e-4:
                break
            Jp = np.zeros((3, s.model.nv), dtype=float)
            Jr = np.zeros((3, s.model.nv), dtype=float)
            mujoco.mj_jac(s.model, s.data, Jp, Jr, np.zeros(3), hand)
            J = Jp[:, :7]
            dq = damped_pinv(J, lam=0.03) @ err
            qdes = s.data.qpos[:7] + gain * dq
            for i, jid in enumerate(dof_idx):
                lo, hi = s.model.jnt_range[jid]
                qdes[i] = np.clip(qdes[i], lo, hi)
            s.data.ctrl[:7] = qdes
            s.data.ctrl[7] = finger_target
            s.step(1)

    for pt in waypoints:
        move_to(pt, gain=0.8, steps=180)

    for _ in range(300):
        s.data.ctrl[:7] = s.data.qpos[:7]
        s.data.ctrl[7] = finger_target
        s.step(1)

    for _ in range(500):
        s.data.ctrl[:7] = s.data.qpos[:7]
        s.data.ctrl[7] = finger_target
        s.step(1)

    peg = s.peg_position()
    # crude align metric
    R = s.data.xmat[s.peg_body_id].reshape(3,3)
    align = abs(R[0, 0])
    return s, peg, align

s, peg, align = run_once()
print('peg', peg)
print('align', align)
insert_x = peg[0]
insertion_progress = max(0.0, min(1.0, (insert_x - 0.47) / (0.515 - 0.47)))
y = abs(peg[1] + 0.10)
z = abs(peg[2] - 0.52)
y_progress = max(0.0, min(1.0, 1.0 - (y - 0.025) / 0.025))
z_progress = max(0.0, min(1.0, 1.0 - (z - 0.020) / 0.020))
alignment_progress = max(0.0, min(1.0, abs(align) / 0.45))
score = 0.45*insertion_progress + 0.15*y_progress + 0.15*z_progress + 0.20*alignment_progress
print('insertion_progress', insertion_progress)
print('y_progress', y_progress)
print('z_progress', z_progress)
print('alignment_progress', alignment_progress)
print('score', score)

s.save_final_state('/work/final_state.npz')
print('saved /work/final_state.npz')
