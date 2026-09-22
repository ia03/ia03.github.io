import numpy as np
import mujoco

from sim import Sim


HAND_BODY_ID = 9  # hand body in the Panda model


def solve_hand_position(sim, target_pos, steps=30, iters=80, step_clip=0.02):
    """Return arm q targets that move the hand body to target_pos."""
    model, data = sim.model, sim.data
    q = data.qpos.copy()
    start = data.xpos[HAND_BODY_ID].copy()
    for s in range(1, steps + 1):
        waypoint = start + (target_pos - start) * (s / steps)
        for _ in range(iters):
            mujoco.mj_forward(model, data)
            pos = data.xpos[HAND_BODY_ID].copy()
            err = waypoint - pos
            if np.linalg.norm(err) < 1e-4:
                break
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, HAND_BODY_ID)
            j = jacp[:, :7]
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(3), err)
            dq = np.clip(dq, -step_clip, step_clip)
            q[:7] += dq
            for joint in range(7):
                lo, hi = model.jnt_range[joint]
                q[joint] = np.clip(q[joint], lo, hi)
            data.qpos[:] = q
            data.qvel[:] = 0
        mujoco.mj_forward(model, data)
    return q[:7].copy()


def hold(sim, arm_q, gripper, n_steps):
    sim.data.ctrl[:7] = arm_q
    sim.data.ctrl[7] = gripper
    sim.step(n_steps)


def main():
    sim = Sim()
    print("start hand", sim.data.xpos[HAND_BODY_ID], "cup", sim.cup_position())

    # First executable end-to-end attempt:
    # - move above cup
    # - descend
    # - close
    # - lift
    # - retreat to robot side
    arm_q = solve_hand_position(sim, np.array([0.55, 0.15, 0.62]))
    hold(sim, arm_q, 0, 40)

    arm_q = solve_hand_position(sim, np.array([0.55, 0.15, 0.53]))
    hold(sim, arm_q, 0, 40)

    hold(sim, arm_q, 255, 120)

    arm_q = solve_hand_position(sim, np.array([0.55, 0.15, 0.64]))
    hold(sim, arm_q, 255, 80)

    arm_q = solve_hand_position(sim, np.array([0.42, 0.15, 0.64]))
    hold(sim, arm_q, 255, 120)

    arm_q = solve_hand_position(sim, np.array([0.35, 0.12, 0.66]))
    hold(sim, arm_q, 255, 120)

    # Let the grasp settle while still holding the cup.
    hold(sim, arm_q, 255, 500)

    print("end hand", sim.data.xpos[HAND_BODY_ID], "cup", sim.cup_position())
    print("contact", sim.has_gripper_cup_contact())
    print("ctrl steps", len(sim._ctrl_trace))
    sim.save_final_state("/work/final_state.npz")
    print("saved /work/final_state.npz")


if __name__ == "__main__":
    main()
