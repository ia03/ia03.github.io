import numpy as np
import mujoco
import imageio.v2 as iio

import sim


MODEL = None
DATA = None
HAND_ID = None
ARM = np.arange(7)
R_TARGET = None


def rotvec_from_R(R):
    tr = np.trace(R)
    c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    ang = np.arccos(c)
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return 0.5 * ang / np.sin(ang) * v


def arm_ik_to(p_target, steps=120, alpha=0.12, orient_weight=0.0):
    global MODEL, DATA, HAND_ID, R_TARGET
    for _ in range(steps):
        mujoco.mj_forward(MODEL, DATA)
        p = DATA.xpos[HAND_ID].copy()
        e_p = p_target - p
        if orient_weight > 0:
            R = DATA.xmat[HAND_ID].reshape(3, 3).copy()
            R_err = R_TARGET @ R.T
            e_r = rotvec_from_R(R_err)
            err = np.concatenate([e_p, orient_weight * e_r])
            jacp = np.zeros((3, MODEL.nv))
            jacr = np.zeros((3, MODEL.nv))
            mujoco.mj_jacBody(MODEL, DATA, jacp, jacr, HAND_ID)
            J = np.vstack([jacp[:, ARM], orient_weight * jacr[:, ARM]])
            dim = 6
        else:
            if np.linalg.norm(e_p) < 1e-3:
                break
            jacp = np.zeros((3, MODEL.nv))
            jacr = np.zeros((3, MODEL.nv))
            mujoco.mj_jacBody(MODEL, DATA, jacp, jacr, HAND_ID)
            J = jacp[:, ARM]
            err = e_p
            dim = 3
        lam = 1e-4
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(dim), err)
        q = DATA.qpos[:7].copy() + alpha * dq
        for i in range(7):
            lo, hi = MODEL.jnt_range[i]
            q[i] = np.clip(q[i], lo + 1e-3, hi - 1e-3)
        DATA.qpos[:7] = q
        DATA.qvel[:] = 0
        mujoco.mj_forward(MODEL, DATA)


def hold(n, arm_ctrl=None, grip_ctrl=None):
    if arm_ctrl is not None:
        DATA.ctrl[:7] = arm_ctrl
    if grip_ctrl is not None:
        DATA.ctrl[7] = grip_ctrl
    sim_obj.step(n)


def settle(n=300):
    # Hold current posture while the physics settles.
    hold(n, arm_ctrl=DATA.qpos[:7].copy(), grip_ctrl=DATA.ctrl[7])


sim_obj = sim.Sim()
MODEL = sim_obj.model
DATA = sim_obj.data
HAND_ID = mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_BODY, "hand")
R_TARGET = DATA.xmat[HAND_ID].reshape(3, 3).copy()

# Start with the gripper open.
DATA.ctrl[:] = 0
DATA.ctrl[7] = 255

# Approach the red block on the branch that has worked well in testing.
waypoints = [
    np.array([0.12, 0.00, 0.85]),
    np.array([0.25, -0.03, 0.75]),
    np.array([0.35, -0.06, 0.65]),
    np.array([0.44, -0.08, 0.58]),
    np.array([0.46, -0.02, 0.56]),
    np.array([0.47, -0.05, 0.54]),
    np.array([0.48, -0.09, 0.52]),
]

for p in waypoints:
    arm_ik_to(p, steps=150, alpha=0.12, orient_weight=0.0)
    hold(100, arm_ctrl=DATA.qpos[:7].copy(), grip_ctrl=255)

# Final grasp pose: lower slightly while staying aligned over the red block.
grasp_pose = np.array([0.48, -0.09, 0.50])
arm_ik_to(grasp_pose, steps=180, alpha=0.10, orient_weight=0.0)
hold(80, arm_ctrl=DATA.qpos[:7].copy(), grip_ctrl=255)

# Close the gripper gradually.
for g in np.linspace(255, 0, 70):
    hold(1, arm_ctrl=DATA.qpos[:7].copy(), grip_ctrl=float(g))

# Lift with the block.
arm_ik_to(np.array([0.48, -0.09, 0.60]), steps=140, alpha=0.10, orient_weight=0.0)
hold(120, arm_ctrl=DATA.qpos[:7].copy(), grip_ctrl=0)

# Move above the green block.
for p in [
    np.array([0.50, -0.04, 0.60]),
    np.array([0.56, 0.00, 0.60]),
    np.array([0.61, 0.05, 0.58]),
    np.array([0.63, 0.08, 0.56]),
]:
    arm_ik_to(p, steps=140, alpha=0.10, orient_weight=0.0)
    hold(80, arm_ctrl=DATA.qpos[:7].copy(), grip_ctrl=0)

# Lower to the stacking height and release.
stack_pose = np.array([0.63, 0.08, 0.50])
arm_ik_to(stack_pose, steps=180, alpha=0.08, orient_weight=0.0)
hold(100, arm_ctrl=DATA.qpos[:7].copy(), grip_ctrl=0)
for g in np.linspace(0, 255, 50):
    hold(1, arm_ctrl=DATA.qpos[:7].copy(), grip_ctrl=float(g))

# Let the stack settle while keeping the arm out of the way.
arm_ik_to(np.array([0.55, 0.00, 0.75]), steps=180, alpha=0.10, orient_weight=0.0)
hold(80, arm_ctrl=DATA.qpos[:7].copy(), grip_ctrl=255)
settle(600)

iio.imwrite("/work/final_attempt.png", sim_obj.render(640, 480))
sim_obj.save_final_state("/work/final_state.npz")

red = sim_obj.block_positions()["red"]
green = sim_obj.block_positions()["green"]
print("final time", DATA.time)
print("red", red)
print("green", green)
print("dx dy dz", red - green)
print("saved /work/final_state.npz")
