import numpy as np
import mujoco

from sim import Sim


HAND_BID = None


def rot_err(current_R, target_R):
    dR = current_R.T @ target_R
    tr = np.trace(dR)
    cosang = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    ang = np.arccos(cosang)
    if ang < 1e-8:
        return np.zeros(3)
    axis = np.array([
        dR[2, 1] - dR[1, 2],
        dR[0, 2] - dR[2, 0],
        dR[1, 0] - dR[0, 1],
    ]) / (2.0 * np.sin(ang))
    return current_R @ (axis * ang)


def solve_ik(model, data, bid, q, target_pos, target_R, arm_joints, max_it=80):
    q = q.copy()
    for _ in range(max_it):
        mujoco.mj_forward(model, data)
        pos = data.xpos[bid].copy()
        R = data.xmat[bid].reshape(3, 3).copy()
        err = np.concatenate([target_pos - pos, 0.5 * rot_err(R, target_R)])
        if np.linalg.norm(err) < 1e-4:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, bid)
        J = np.vstack([jacp[:, arm_joints], 0.5 * jacr[:, arm_joints]])
        lam = 1e-2
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
        q[:7] += dq
        for i in arm_joints:
            lo, hi = model.jnt_range[i]
            q[i] = np.clip(q[i], lo, hi)
        data.qpos[:9] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
    return q


def move_interp(sim, target_qpos, steps, grip_ctrl=None):
    start = sim.data.qpos[:9].copy()
    for t in range(steps):
        a = (t + 1) / steps
        q = (1 - a) * start + a * target_qpos
        sim.data.ctrl[:7] = q[:7]
        sim.data.ctrl[7] = 255 if grip_ctrl is None else grip_ctrl
        sim.step()


def main():
    sim = Sim()
    model, data = sim.model, sim.data
    hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    arm_joints = np.arange(7)

    home = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04])

    # Get to the known home pose first.
    move_interp(sim, home, 800, grip_ctrl=255)

    mujoco.mj_forward(model, data)
    target_R = data.xmat[hand_bid].reshape(3, 3).copy()

    q = data.qpos[:9].copy()
    hover = solve_ik(model, data, hand_bid, q, np.array([0.5, 0.0, 0.56]), target_R, arm_joints)
    hover[7:] = 0.04
    move_interp(sim, hover, 500, grip_ctrl=255)

    q = data.qpos[:9].copy()
    grasp = solve_ik(model, data, hand_bid, q, np.array([0.5, 0.0, 0.525]), target_R, arm_joints)
    grasp[7:] = 0.04
    move_interp(sim, grasp, 350, grip_ctrl=255)

    # Close the gripper while staying on target.
    for _ in range(350):
        sim.data.ctrl[:7] = grasp[:7]
        sim.data.ctrl[7] = 0
        sim.step()

    # Lift while keeping the cup squeezed.
    q = data.qpos[:9].copy()
    lift = solve_ik(model, data, hand_bid, q, np.array([0.5, 0.0, 0.60]), target_R, arm_joints)
    lift[7:] = 0.0
    move_interp(sim, lift, 700, grip_ctrl=0)

    # Hold long enough to make the final replay settle robust.
    for _ in range(700):
        sim.data.ctrl[:7] = lift[:7]
        sim.data.ctrl[7] = 0
        sim.step()

    print('final hand', sim.data.xpos[hand_bid], 'cup', sim.cup_position(), 'finger qpos', sim.data.qpos[7:9])
    sim.save_final_state('/work/final_state.npz')
    print('saved /work/final_state.npz')


if __name__ == '__main__':
    main()
