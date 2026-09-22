import time
import numpy as np
import mujoco

from sim import Sim


START_TS = time.time()
DEADLINE_TS = START_TS + 14 * 60


def utc_now():
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def print_budget(label):
    remaining = DEADLINE_TS - time.time()
    print(f"[{utc_now()}] {label}; remaining ~{remaining:.1f}s")


sim = Sim()
m = sim.model
d = sim.data

hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
left_finger_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
right_finger_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
R_HOME = np.array(d.xmat[hand_id]).reshape(3, 3).copy()
TIP_OFFSET_LOCAL = np.array([0.0, 0.0, 0.1029])
ARM_LIMITS = m.actuator_ctrlrange[:7].copy()

best_score = -1.0


def orientation_error(r_des, r_cur):
    return 0.5 * (
        np.cross(r_cur[:, 0], r_des[:, 0])
        + np.cross(r_cur[:, 1], r_des[:, 1])
        + np.cross(r_cur[:, 2], r_des[:, 2])
    )


def hand_target_for_tip(target_tip):
    return np.asarray(target_tip) + R_HOME @ TIP_OFFSET_LOCAL


def evaluate():
    block = sim.block_position()
    drawer = sim.drawer_open_amount()
    contact = float(sim.has_gripper_block_contact())
    return {
        "drawer": drawer,
        "block_z": block[2],
        "block_x": block[0],
        "contact": contact,
        "score": 2.0 * contact + 4.0 * max(0.0, drawer - 0.05) + 6.0 * max(0.0, block[2] - 0.43),
    }


def save_if_best(tag):
    global best_score
    metrics = evaluate()
    if metrics["score"] > best_score:
        best_score = metrics["score"]
        sim.save_final_state("/work/final_state.npz")
        print(f"saved {tag}: {metrics}")
    else:
        print(f"kept prior save at {tag}: {metrics}")


def set_ctrl(q_arm=None, grip=None, drawer=None):
    if q_arm is not None:
        d.ctrl[:7] = q_arm
    if grip is not None:
        d.ctrl[7] = grip
    if drawer is not None:
        d.ctrl[8] = drawer


def solve_ik(hand_target, q_seed, orient_weight=0.03, iters=60):
    q = np.clip(np.asarray(q_seed, dtype=float), ARM_LIMITS[:, 0], ARM_LIMITS[:, 1]).copy()
    d.qpos[:7] = q
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(iters):
        hand_pos = np.array(d.xpos[hand_id])
        r_cur = np.array(d.xmat[hand_id]).reshape(3, 3)
        err_p = np.asarray(hand_target) - hand_pos
        err_o = orientation_error(R_HOME, r_cur)
        err = np.concatenate([err_p, orient_weight * err_o])
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, jacr, hand_id)
        j = np.vstack([jacp[:, :7], orient_weight * jacr[:, :7]])
        dq = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(6), err)
        q = np.clip(d.qpos[:7] + dq, ARM_LIMITS[:, 0], ARM_LIMITS[:, 1])
        d.qpos[:7] = q
        mujoco.mj_forward(m, d)
    return q


def move_tip_to(target_tip, steps, grip, drawer=1.0, orient_weight=0.03):
    hand_target = hand_target_for_tip(target_tip)
    q_target = d.qpos[:7].copy()
    for i in range(steps):
        q_target = solve_ik(hand_target, q_target, orient_weight=orient_weight, iters=8)
        set_ctrl(q_target, grip=grip, drawer=drawer)
        sim.step(1)
        if i % 50 == 0:
            block = sim.block_position()
            print(
                f"  step {i:4d} hand={np.round(d.xpos[hand_id],4)} block={np.round(block,4)} "
                f"drawer={sim.drawer_open_amount():.3f} contact={sim.has_gripper_block_contact()}"
            )


def hold(steps, grip, drawer=1.0):
    q_hold = d.ctrl[:7].copy()
    set_ctrl(q_hold, grip=grip, drawer=drawer)
    sim.step(steps)


print_budget("start")

# Stage 1: open drawer and save a valid early trajectory.
set_ctrl(np.zeros(7), grip=255, drawer=1.0)
sim.step(220)
print("after open", evaluate(), sim.block_position())
save_if_best("drawer-open")

# Stage 2: reach above the presented block.
block = sim.block_position().copy()
approach_tip = np.array([block[0], block[1], block[2] + 0.12])
move_tip_to(approach_tip, steps=220, grip=255, drawer=1.0, orient_weight=0.05)

# Stage 3: descend around the block.
block = sim.block_position().copy()
grasp_tip = np.array([block[0], block[1], block[2] + 0.015])
move_tip_to(grasp_tip, steps=220, grip=255, drawer=1.0, orient_weight=0.05)
save_if_best("pre-close")

# Stage 4: close fingers and let contact settle.
for grip in [180, 120, 80, 40, 0]:
    hold(70, grip=grip, drawer=1.0)
    print("close", grip, evaluate(), np.round(sim.block_position(), 4))
save_if_best("after-close")

# Stage 5: lift while holding the drawer open.
for z in [0.56, 0.62, 0.70, 0.78]:
    block = sim.block_position().copy()
    lift_tip = np.array([block[0], block[1], z])
    move_tip_to(lift_tip, steps=180, grip=0, drawer=1.0, orient_weight=0.05)
    hold(40, grip=0, drawer=1.0)
    print("lift", z, evaluate(), np.round(sim.block_position(), 4))
    save_if_best(f"lift-{z:.2f}")

# Stage 6: retreat slightly back from the drawer and hold for settle.
block = sim.block_position().copy()
carry_tip = np.array([min(block[0] - 0.05, 0.72), block[1], max(0.72, block[2] + 0.08)])
move_tip_to(carry_tip, steps=220, grip=0, drawer=1.0, orient_weight=0.05)
hold(700, grip=0, drawer=1.0)
print("final", evaluate(), np.round(sim.block_position(), 4), "drawer", sim.drawer_open_amount())
save_if_best("final")
print_budget("done")
