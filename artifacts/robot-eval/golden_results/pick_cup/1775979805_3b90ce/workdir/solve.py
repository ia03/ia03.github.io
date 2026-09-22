"""Pick up the cup - direct PD controller lift without interpolation."""
import numpy as np
import mujoco
from sim import Sim
from PIL import Image
from scipy.optimize import minimize

sim = Sim()
m = sim.model
d = sim.data

hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
lf_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
rf_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

bounds = [(m.jnt_range[i, 0], m.jnt_range[i, 1]) for i in range(7)]

def fk_full(q, fo=0.04):
    d_tmp = mujoco.MjData(m)
    d_tmp.qpos[:7] = q
    d_tmp.qpos[7:9] = [fo, fo]
    mujoco.mj_forward(m, d_tmp)
    return (d_tmp.xpos[hand_id].copy(),
            d_tmp.xmat[hand_id].reshape(3,3).copy(),
            d_tmp.xpos[lf_id].copy(),
            d_tmp.xpos[rf_id].copy())

def find_config(target_fz, q_seed=None):
    """Find config targeting finger midpoint at (0.5, 0, target_fz) with hand z-axis down."""
    def cost(q):
        hp, hr, lf, rf = fk_full(q)
        fm = (lf + rf) / 2
        pos_err = np.array([fm[0] - 0.5, fm[1], fm[2] - target_fz])
        orient_err = hr[:, 2] - np.array([0, 0, -1])
        return np.sum(pos_err**2) * 10 + np.sum(orient_err**2)

    seeds = [
        [0, 0.3, 0, -1.5, 0, 1.8, 0],
        [0, 0.5, 0, -1.5, 0, 2.0, 0],
        [0, -0.3, 0, -1.5, 0, 1.8, 0],
        [0, 0.3, 0, -2.0, 0, 2.3, 0],
    ]
    if q_seed is not None:
        seeds.insert(0, list(q_seed))

    best_q, best_cost = None, float('inf')
    for seed in seeds:
        res = minimize(cost, seed, method='L-BFGS-B', bounds=bounds)
        if res.fun < best_cost:
            best_cost = res.fun
            best_q = res.x.copy()
    return best_q

def render_save(name):
    img = sim.render(width=640, height=480)
    Image.fromarray(img).save(f"/work/{name}.png")

# === Find key configs ===
q_above = find_config(0.535)  # fingers 10cm above cup center
q_grasp = find_config(0.435)  # fingers at cup center height
q_lift  = find_config(0.72, q_seed=q_grasp)   # lifted position

# Verify configs
for name, q in [('above', q_above), ('grasp', q_grasp), ('lift', q_lift)]:
    hp, hr, lf, rf = fk_full(q)
    fm = (lf + rf) / 2
    print(f"{name}: q={np.round(q, 4)}, finger_mid={np.round(fm, 4)}, z_axis={np.round(hr[:,2], 3)}")

# === Execute ===
print("\n--- Execute ---")

# 1. Open gripper
d.ctrl[7] = 255
sim.step(500)
print(f"1. Cup={sim.cup_position()}")

# 2. Move above - just set target and wait
d.ctrl[:7] = q_above
sim.step(5000)  # plenty of time for PD to converge
cup = sim.cup_position()
print(f"2. Above. Cup={cup}, Hand={d.xpos[hand_id]}")
render_save("step2")

# 3. Descend - set grasp target and wait
d.ctrl[:7] = q_grasp
sim.step(5000)
cup = sim.cup_position()
print(f"3. Grasp pos. Cup={cup}")
print(f"   LF={d.xpos[lf_id]}, RF={d.xpos[rf_id]}")
render_save("step3")

# 4. Close gripper
d.ctrl[7] = 0
sim.step(3000)
cup = sim.cup_position()
fj = d.qpos[7:9].copy()
print(f"4. Closed. Cup={cup}, FJ={fj}")
render_save("step4")

# 5. Lift - just set target and wait (PD controller handles smooth motion)
d.ctrl[:7] = q_lift
sim.step(10000)  # lots of time to reach target smoothly
cup = sim.cup_position()
print(f"5. Lifted. Cup={cup}")
render_save("step5")

if cup[2] < 0.5:
    print("Cup lost!")
    exit(1)

# 6. Settle more
sim.step(5000)
cup = sim.cup_position()
print(f"6. Settled. Cup={cup}")

# 7. Zero velocities and verify
d.qvel[:] = 0.0
mujoco.mj_forward(m, d)
qpos_s = d.qpos.copy()
qvel_s = d.qvel.copy()
ctrl_s = d.ctrl.copy()

print("\n--- Verify ---")
for t_s in [0.5, 1.0, 1.5, 2.0, 2.5]:
    d.qpos[:] = qpos_s; d.qvel[:] = qvel_s; d.ctrl[:] = ctrl_s
    mujoco.mj_forward(m, d)
    sim.step(int(t_s / m.opt.timestep))
    cz = sim.cup_position()[2]
    print(f"  t={t_s:.1f}s: cup_z={cz:.4f} {'PASS' if cz >= 0.55 else 'FAIL'}")

# Check final
d.qpos[:] = qpos_s; d.qvel[:] = qvel_s; d.ctrl[:] = ctrl_s
mujoco.mj_forward(m, d)
sim.step(1250)
cz_final = sim.cup_position()[2]

if cz_final >= 0.55:
    d.qpos[:] = qpos_s; d.qvel[:] = qvel_s; d.ctrl[:] = ctrl_s
    mujoco.mj_forward(m, d)
    sim.save_final_state("/work/final_state.npz")
    print(f"\nSUCCESS! Cup z after 2.5s = {cz_final:.4f}")
else:
    print(f"\nFAIL: {cz_final:.4f}")
