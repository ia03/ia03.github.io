"""Pick up the cup - pad IK for both grasp and lift."""
import numpy as np
import mujoco
from sim import Sim
from PIL import Image

sim = Sim()
m = sim.model
d = sim.data

hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
lf_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
rf_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
cup_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_geom")

# Collision filtering
m.geom_contype[cup_geom_id] = 2
m.geom_conaffinity[cup_geom_id] = 2
for gname in ["table_top", "floor"]:
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
    m.geom_contype[gid] = 3
    m.geom_conaffinity[gid] = 3
for i in range(m.ngeom):
    bid = m.geom_bodyid[i]
    bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
    if bname in ("hand", "link7"):
        m.geom_contype[i] = 1
        m.geom_conaffinity[i] = 1
    elif bname in ("left_finger", "right_finger"):
        m.geom_contype[i] = 3
        m.geom_conaffinity[i] = 3

LEFT_PAD = 69
RIGHT_PAD = 77

def pad_center():
    return 0.5 * (d.geom_xpos[LEFT_PAD] + d.geom_xpos[RIGHT_PAD])

def pad_jac():
    jacp_l = np.zeros((3, m.nv))
    jacp_r = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jacp_l, None, d.geom_xpos[LEFT_PAD], lf_id)
    mujoco.mj_jac(m, d, jacp_r, None, d.geom_xpos[RIGHT_PAD], rf_id)
    return 0.5 * (jacp_l[:, :7] + jacp_r[:, :7])

def frame(name):
    Image.fromarray(sim.render(640, 480)).save(f'/work/{name}.png')

def ik_pads(target, n_iters=300, substeps=20, alpha=0.5):
    for i in range(n_iters):
        mujoco.mj_forward(m, d)
        J = pad_jac()
        err = target - pad_center()
        if np.linalg.norm(err) < 0.001:
            break
        dq = alpha * J.T @ np.linalg.solve(J @ J.T + 0.005*np.eye(3), err)
        q = d.qpos[:7] + dq
        for j in range(7):
            q[j] = np.clip(q[j], m.jnt_range[j,0]+0.005, m.jnt_range[j,1]-0.005)
        d.ctrl[:7] = q
        sim.step(n=substeps)
    mujoco.mj_forward(m, d)
    return np.linalg.norm(target - pad_center())

def cup_contacts():
    names = set()
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 == cup_geom_id or c.geom2 == cup_geom_id:
            other = c.geom2 if c.geom1 == cup_geom_id else c.geom1
            names.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[other]))
    return names

cup_start = sim.cup_position().copy()
print(f"Cup: {cup_start}")

# Step 0: Safe start
d.ctrl[:7] = [0.0, -0.5, 0.0, -2.5, 0.0, 2.5, 0.7854]
d.ctrl[7] = 255
sim.step(n=3000)
mujoco.mj_forward(m, d)

# Step 1: Pads above cup
err = ik_pads(np.array([cup_start[0], cup_start[1], cup_start[2]+0.10]),
              n_iters=500, alpha=0.5)
sim.step(n=1500)
mujoco.mj_forward(m, d)
print(f"Above: pads={pad_center()}, err={err:.4f}")

# Step 2: Pads to cup center
err = ik_pads(cup_start.copy(), n_iters=600, substeps=15, alpha=0.3)
sim.step(n=2000)
mujoco.mj_forward(m, d)
print(f"At cup: pads={pad_center()}, cup={sim.cup_position()}")

# Step 3: Close gripper
d.ctrl[7] = 0
sim.step(n=4000)
mujoco.mj_forward(m, d)
cup_now = sim.cup_position()
print(f"Closed: cup={cup_now}, fj={d.qpos[7:9]}, contacts={cup_contacts()}")
sim.save_final_state("/work/final_state.npz")
frame("grasp")

# Step 4: Lift using pad IK - KEEP PADS CENTERED ON CUP while lifting
print("\n=== Lifting with pad IK ===")
pad_now = pad_center().copy()
cup_now = sim.cup_position()

# Lift in very small Z increments, keeping X and Y constant
z_targets = np.linspace(pad_now[2], pad_now[2] + 0.20, 100)

for i, zt in enumerate(z_targets):
    target = np.array([pad_now[0], pad_now[1], zt])
    # Use pad IK to move pads to target
    for _ in range(10):
        mujoco.mj_forward(m, d)
        J = pad_jac()
        err = target - pad_center()
        if np.linalg.norm(err) < 0.001:
            break
        dq = 0.3 * J.T @ np.linalg.solve(J @ J.T + 0.005*np.eye(3), err)
        q = d.qpos[:7] + dq
        for j in range(7):
            q[j] = np.clip(q[j], m.jnt_range[j,0]+0.005, m.jnt_range[j,1]-0.005)
        d.ctrl[:7] = q
        d.ctrl[7] = 0  # keep closed
        sim.step(n=10)

    if i % 20 == 0:
        mujoco.mj_forward(m, d)
        c = sim.cup_position()
        contacts = cup_contacts()
        print(f"  Step {i}: pad={pad_center()}, cup_z={c[2]:.4f}, contacts={contacts}")
        if c[2] >= 0.52:
            print("  CUP LIFTED!")
            sim.save_final_state("/work/final_state.npz")

# Final settle
sim.step(n=2000)
mujoco.mj_forward(m, d)
cup_final = sim.cup_position()
print(f"\nFinal: cup={cup_final}, z={cup_final[2]:.4f}")
frame("lifted")

if cup_final[2] >= 0.52:
    print("SUCCESS!")

sim.save_final_state("/work/final_state.npz")
print("Saved")
