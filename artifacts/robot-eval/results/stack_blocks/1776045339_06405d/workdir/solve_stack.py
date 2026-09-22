import numpy as np
import mujoco

from sim import Sim


def rot_err(current, desired):
    return 0.5 * (
        np.cross(current[:, 0], desired[:, 0])
        + np.cross(current[:, 1], desired[:, 1])
        + np.cross(current[:, 2], desired[:, 2])
    )


def main():
    sim = Sim()
    model = sim.model
    data = sim.data

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = sim.red_block_id
    green_id = sim.green_block_id
    home = model.key_qpos[0, :7].copy()

    def solve_ik(target_pos, target_rot, q_init, iters=200):
        q = q_init.copy()
        qpos_base = data.qpos.copy()
        qvel_base = data.qvel.copy()
        for _ in range(iters):
            data.qpos[:] = qpos_base
            data.qvel[:] = qvel_base
            data.qpos[:7] = q
            mujoco.mj_forward(model, data)

            pos = data.xpos[hand_id].copy()
            rot = data.xmat[hand_id].reshape(3, 3).copy()
            err = np.concatenate([target_pos - pos, 0.5 * rot_err(rot, target_rot)])

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
            jac = np.vstack([jacp[:, :7], jacr[:, :7]])
            dq = np.linalg.solve(jac.T @ jac + 1e-4 * np.eye(7), jac.T @ err)
            q += 0.8 * dq
            q = np.clip(q, model.jnt_range[:7, 0], model.jnt_range[:7, 1])

            if np.linalg.norm(err[:3]) < 1e-5 and np.linalg.norm(err[3:]) < 1e-4:
                break

        data.qpos[:] = qpos_base
        data.qvel[:] = qvel_base
        mujoco.mj_forward(model, data)
        return q

    def command(q, grip, steps):
        for _ in range(steps):
            data.ctrl[:7] = q
            data.ctrl[7] = 255 * (grip / 0.04)
            sim.step(1)

    def move(q_from, q_to, grip, steps):
        for alpha in np.linspace(0.0, 1.0, steps):
            data.ctrl[:7] = (1.0 - alpha) * q_from + alpha * q_to
            data.ctrl[7] = 255 * (grip / 0.04)
            sim.step(1)

    command(home, 0.04, 600)
    desired_rot = data.xmat[hand_id].reshape(3, 3).copy()
    q_home = data.qpos[:7].copy()

    red0 = data.xpos[red_id].copy()
    green0 = data.xpos[green_id].copy()
    fingertip_midpoint_offset = np.array([8.2473e-4, 0.0, 5.8394e-2])

    pre_red = solve_ik(red0 + fingertip_midpoint_offset + np.array([0.0, 0.0, 0.10]), desired_rot, q_home)
    grasp = solve_ik(red0 + fingertip_midpoint_offset, desired_rot, pre_red)
    lift = solve_ik(red0 + fingertip_midpoint_offset + np.array([0.0, 0.0, 0.12]), desired_rot, grasp)

    move(q_home, pre_red, 0.04, 400)
    move(pre_red, grasp, 0.04, 300)
    command(grasp, 0.04, 120)
    for grip in np.linspace(0.04, 0.0, 240):
        data.ctrl[:7] = grasp
        data.ctrl[7] = 255 * (grip / 0.04)
        sim.step(1)
    command(grasp, 0.0, 200)
    move(grasp, lift, 0.0, 500)
    command(lift, 0.0, 150)

    hold_offset = data.xpos[hand_id].copy() - data.xpos[red_id].copy()
    stack_center = data.xpos[green_id].copy() + np.array([0.0, 0.0, 0.055])

    pre_green = solve_ik(stack_center + hold_offset + np.array([0.0, 0.0, 0.10]), desired_rot, data.qpos[:7].copy())
    place_hi = solve_ik(stack_center + hold_offset + np.array([0.0, 0.0, 0.02]), desired_rot, pre_green)
    place = solve_ik(stack_center + hold_offset, desired_rot, place_hi)
    retreat = solve_ik(stack_center + hold_offset + np.array([0.0, 0.0, 0.12]), desired_rot, place)

    move(lift, pre_green, 0.0, 600)
    move(pre_green, place_hi, 0.0, 300)
    move(place_hi, place, 0.0, 240)
    command(place, 0.0, 160)
    for grip in np.linspace(0.0, 0.04, 260):
        data.ctrl[:7] = place
        data.ctrl[7] = 255 * (grip / 0.04)
        sim.step(1)
    command(place, 0.04, 300)
    move(place, retreat, 0.04, 300)
    command(retreat, 0.04, 200)
    command(home, 0.04, 1000)

    sim.save_final_state("/work/final_state.npz")

    red = data.xpos[red_id].copy()
    green = data.xpos[green_id].copy()
    print("final red:", red)
    print("final green:", green)
    print("final diff:", red - green)
    print("steps:", len(sim._ctrl_trace))


if __name__ == "__main__":
    main()
