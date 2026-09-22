
import numpy as np
import mujoco
from sim import Sim

HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
SAFE = np.array([0.0, -0.35, 0.0, -1.90, 0.0, 1.95, 0.785], dtype=float)
Q_HANDLE = np.array([0.260114734942896, 0.5978935416914477, -0.2548739609310991, -1.7011568239308756, 0.0731417910040021, 3.7525, 0.8723261334629767], dtype=float)
Q_PULL = np.array([0.12598874856351472, 1.1973462499176968, -0.34850701301211456, -0.625005200213342, -0.022487162744560703, 3.3513450345879296, 1.1390361931507589], dtype=float)
DOWN = np.array([0.0, 0.0, -1.0], dtype=float)
TILT_Y_PLUS = np.array([0.0, 0.10, -0.995], dtype=float)
TILT_Y_MINUS = np.array([0.0, -0.10, -0.995], dtype=float)
TILT_X_PLUS = np.array([0.10, 0.0, -0.995], dtype=float)
TILT_X_MINUS = np.array([-0.10, 0.0, -0.995], dtype=float)

def solve_midpoint_ik(model, data, hand_id, left_id, right_id, target_midpoint, seed, target_down):
    q = seed.copy()
    qmin = model.actuator_ctrlrange[:7, 0]
    qmax = model.actuator_ctrlrange[:7, 1]
    target_down = np.array(target_down, dtype=float)
    target_down = target_down / max(np.linalg.norm(target_down), 1e-9)
    for _ in range(160):
        data.qpos[:7] = q
        data.qpos[7:9] = 0.04
        mujoco.mj_forward(model, data)
        hand_z = data.xmat[hand_id].reshape(3, 3)[:, 2]
        midpoint = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
        pos_err = target_midpoint - midpoint
        ori_err = np.cross(hand_z, target_down)
        jacp_l = np.zeros((3, model.nv)); jacr_l = np.zeros((3, model.nv))
        jacp_r = np.zeros((3, model.nv)); jacr_r = np.zeros((3, model.nv))
        jacp_h = np.zeros((3, model.nv)); jacr_h = np.zeros((3, model.nv))
        mujoco.mj_jacBodyCom(model, data, jacp_l, jacr_l, left_id)
        mujoco.mj_jacBodyCom(model, data, jacp_r, jacr_r, right_id)
        mujoco.mj_jacBody(model, data, jacp_h, jacr_h, hand_id)
        jac = np.vstack([0.5 * (jacp_l[:, :7] + jacp_r[:, :7]), 0.35 * jacr_h[:, :7]])
        err = np.concatenate([pos_err, 0.35 * ori_err])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), err)
        q = np.clip(q + 0.8 * dq, qmin, qmax)
    return q

def move_q(sim, q_target, grip, steps, drawer_ctrl=1.0):
    q_start = sim.data.qpos[:7].copy()
    g_start = float(sim.data.ctrl[7])
    d_start = float(sim.data.ctrl[8])
    for i in range(steps):
        a = (i + 1) / steps
        sim.data.ctrl[:7] = (1 - a) * q_start + a * q_target
        sim.data.ctrl[7] = (1 - a) * g_start + a * grip
        sim.data.ctrl[8] = (1 - a) * d_start + a * drawer_ctrl
        sim.step(1)

def hold(sim, q_target, grip, steps, drawer_ctrl=1.0):
    for _ in range(steps):
        sim.data.ctrl[:7] = q_target
        sim.data.ctrl[7] = grip
        sim.data.ctrl[8] = drawer_ctrl
        sim.step(1)

def best_q(model, data, hand_id, left_id, right_id, target_xyz, target_down, seeds, extra_seed=None):
    target = np.array(target_xyz, dtype=float)
    candidates = []
    all_seeds = list(seeds)
    if extra_seed is not None:
        all_seeds.append(extra_seed.copy())
    for seed in all_seeds:
        q = solve_midpoint_ik(model, data, hand_id, left_id, right_id, target, seed, target_down)
        data.qpos[:7] = q
        data.qpos[7:9] = 0.04
        mujoco.mj_forward(model, data)
        midpoint = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
        hand_z = data.xmat[hand_id].reshape(3, 3)[:, 2]
        err = np.linalg.norm(midpoint - target) + 0.2 * np.linalg.norm(np.cross(hand_z, target_down / max(np.linalg.norm(target_down), 1e-9)))
        candidates.append((err, q))
    return min(candidates, key=lambda item: item[0])[1]

def open_drawer_motor(sim):
    sim.data.ctrl[:7] = HOME
    sim.data.ctrl[7] = 255
    sim.data.ctrl[8] = 1.0
    sim.step(320)
    hold(sim, HOME, 255, 80, drawer_ctrl=1.0)
    hold(sim, SAFE, 255, 60, drawer_ctrl=1.0)

def attempt(sim, target_down, xy_offset, grasp_z_extra, approach_z_extra, lift_z):
    model = sim.model
    data = sim.data
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_finger')
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_finger')
    block = sim.block_position().copy()
    bx = float(block[0] + xy_offset[0])
    by = float(block[1] + xy_offset[1])
    bz = float(block[2])
    seeds = [HOME.copy(), np.array([0.08, -0.19, -0.07, -1.88, -0.01, 1.69, 0.5]), np.array([-0.08, -0.19, 0.07, -1.88, 0.01, 1.69, -0.5]), Q_HANDLE.copy(), Q_PULL.copy(), SAFE.copy()]
    above_far = np.array([bx, by, max(bz + approach_z_extra, 0.24)], dtype=float)
    above = np.array([bx, by, max(bz + 0.14, 0.12)], dtype=float)
    grasp = np.array([bx, by, max(bz + grasp_z_extra, 0.065)], dtype=float)
    lift = np.array([bx, by, lift_z], dtype=float)
    q_above_far = best_q(model, data, hand_id, left_id, right_id, above_far, target_down, seeds, extra_seed=SAFE)
    q_above = best_q(model, data, hand_id, left_id, right_id, above, target_down, seeds, extra_seed=q_above_far)
    q_grasp = best_q(model, data, hand_id, left_id, right_id, grasp, target_down, seeds, extra_seed=q_above)
    q_lift = best_q(model, data, hand_id, left_id, right_id, lift, target_down, seeds, extra_seed=q_grasp)
    move_q(sim, q_above_far, 255, 180, drawer_ctrl=1.0)
    move_q(sim, q_above, 255, 120, drawer_ctrl=1.0)
    move_q(sim, q_grasp, 255, 100, drawer_ctrl=1.0)
    move_q(sim, q_grasp, 80, 120, drawer_ctrl=1.0)
    move_q(sim, q_grasp, 0, 160, drawer_ctrl=1.0)
    hold(sim, q_grasp, 0, 120, drawer_ctrl=1.0)
    move_q(sim, q_lift, 0, 240, drawer_ctrl=1.0)
    hold(sim, q_lift, 0, 280, drawer_ctrl=1.0)
    return float(sim.block_position()[2]), bool(sim.has_gripper_block_contact()), sim.drawer_open_amount(), q_lift

sim = Sim()
open_drawer_motor(sim)
print('after open', sim.drawer_open_amount(), sim.block_position())

best = None
for td_name, td in [('DOWN', DOWN), ('TY+', TILT_Y_PLUS), ('TY-', TILT_Y_MINUS), ('TX+', TILT_X_PLUS), ('TX-', TILT_X_MINUS), ('mix1', np.array([0.07, 0.07, -0.995])), ('mix2', np.array([-0.07, 0.07, -0.995]))]:
    for xy_off in [(0.0, 0.0), (0.0, 0.008), (0.0, -0.008), (0.008, 0.0), (-0.008, 0.0), (0.012, 0.0), (-0.012, 0.0)]:
        for grasp_z in [0.035, 0.045, 0.055, 0.065, 0.075]:
            for lift_z in [0.58, 0.60, 0.62, 0.64, 0.66, 0.68]:
                try:
                    z, contact, drawer, q = attempt(sim, td, xy_off, grasp_z, 0.22, lift_z)
                except Exception as exc:
                    print('ERR', td_name, xy_off, grasp_z, lift_z, exc)
                    raise
                print(f"{td_name} off={xy_off} grasp={grasp_z:.3f} lift={lift_z:.2f} -> z={z:.3f} contact={contact} drawer={drawer:.3f}")
                if best is None or z > best[0] or (z == best[0] and contact and not best[1]):
                    best = (z, contact, td_name, xy_off, grasp_z, lift_z, drawer)
                if z >= 0.60 and contact and drawer >= 0.05:
                    print('SUCCESS', td_name, xy_off, grasp_z, lift_z, z, drawer)
                    sim.save_final_state('/work/final_state.npz')
                    raise SystemExit(0)
                sim.reset()
                open_drawer_motor(sim)

print('BEST', best)
sim.save_final_state('/work/final_state.npz')
