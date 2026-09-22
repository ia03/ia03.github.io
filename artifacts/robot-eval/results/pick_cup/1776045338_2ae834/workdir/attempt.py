import argparse
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


@dataclass
class EvalResult:
    final_cup_z: float
    max_cup_z: float
    contact_fraction: float
    progress_score: float


class PandaCupController:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.hand_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.left_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self.cup_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cup_geom")
        self.arm_range = self.model.actuator_ctrlrange[:7].copy()
        self.gripper_range = self.model.actuator_ctrlrange[7].copy()
        self.home_rot = self.data.xmat[self.hand_bid].reshape(3, 3).copy()
        self._finger_geom_ids = {
            i
            for i in range(self.model.ngeom)
            if self.model.geom_bodyid[i] in (self.left_bid, self.right_bid)
        }

    def grip_center(self):
        return 0.5 * (self.data.xpos[self.left_bid] + self.data.xpos[self.right_bid])

    def grip_jacobian(self):
        jacp_l = np.zeros((3, self.model.nv))
        jacp_r = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp_l, None, self.left_bid)
        mujoco.mj_jacBody(self.model, self.data, jacp_r, None, self.right_bid)
        return 0.5 * (jacp_l[:, :7] + jacp_r[:, :7])

    def hand_rot_jacobian(self):
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, None, jacr, self.hand_bid)
        return jacr[:, :7]

    def hand_rot(self):
        return self.data.xmat[self.hand_bid].reshape(3, 3).copy()

    def cup_contact(self):
        left, right = self.contact_sides()
        return left or right

    def contact_sides(self):
        left = False
        right = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.geom1 == self.cup_gid:
                body = self.model.geom_bodyid[con.geom2]
            elif con.geom2 == self.cup_gid:
                body = self.model.geom_bodyid[con.geom1]
            else:
                continue
            if body == self.left_bid:
                left = True
            if body == self.right_bid:
                right = True
        return left, right

    def move_stage(
        self,
        target_pos,
        steps,
        gripper,
        pos_gain=1.8,
        damp=1e-2,
        max_joint_step=0.05,
        target_rot=None,
        rot_gain=0.6,
    ):
        target_pos = np.array(target_pos, dtype=float)
        for _ in range(steps):
            pos = self.grip_center()
            err = target_pos - pos
            jac = self.grip_jacobian()
            rhs = pos_gain * err
            if target_rot is not None:
                cur_rot = self.hand_rot()
                rot_err = 0.5 * (
                    np.cross(cur_rot[:, 0], target_rot[:, 0])
                    + np.cross(cur_rot[:, 1], target_rot[:, 1])
                    + np.cross(cur_rot[:, 2], target_rot[:, 2])
                )
                jac = np.vstack([jac, self.hand_rot_jacobian()])
                rhs = np.concatenate([rhs, rot_gain * rot_err])
                reg = damp * np.eye(6)
            else:
                reg = damp * np.eye(3)
            a = jac @ jac.T + reg
            dq = jac.T @ np.linalg.solve(a, rhs)
            dq = np.clip(dq, -max_joint_step, max_joint_step)
            q_target = np.clip(self.data.qpos[:7] + dq, self.arm_range[:, 0], self.arm_range[:, 1])
            self.data.ctrl[:7] = q_target
            self.data.ctrl[7] = float(np.clip(gripper, self.gripper_range[0], self.gripper_range[1]))
            self.sim.step()


def replay_eval(ctrl_trace):
    sim = Sim()
    controller = PandaCupController(sim)
    max_cup_z = sim.cup_position()[2]
    for ctrl in ctrl_trace:
        sim.data.ctrl[:] = ctrl
        sim.step()
        max_cup_z = max(max_cup_z, sim.cup_position()[2])
    contact_count = 0
    for _ in range(500):
        sim.step()
        max_cup_z = max(max_cup_z, sim.cup_position()[2])
        if controller.cup_contact():
            contact_count += 1
    final_cup_z = float(sim.cup_position()[2])
    contact_fraction = contact_count / 500.0
    height_progress = np.clip((final_cup_z - 0.435) / (0.52 - 0.435), 0.0, 1.0)
    contact_progress = np.clip(contact_fraction / 0.5, 0.0, 1.0)
    progress_score = 0.5 * height_progress + 0.5 * contact_progress
    return EvalResult(final_cup_z, max_cup_z, contact_fraction, float(progress_score))


def build_trace(args):
    sim = Sim()
    controller = PandaCupController(sim)

    controller.move_stage(
        [0.50, 0.0, 0.62],
        steps=args.approach_steps,
        gripper=args.open_gripper,
        target_rot=controller.home_rot,
        rot_gain=args.rot_gain,
    )
    controller.move_stage(
        [0.50, 0.0, args.pregrasp_z],
        steps=args.pregrasp_steps,
        gripper=args.open_gripper,
        target_rot=controller.home_rot,
        rot_gain=args.rot_gain,
    )

    close_schedule = np.linspace(args.open_gripper, args.close_gripper, args.close_steps)
    for g in close_schedule:
        controller.move_stage(
            [args.grasp_x, 0.0, args.grasp_z],
            steps=1,
            gripper=float(g),
            pos_gain=args.close_gain,
            damp=args.close_damp,
            max_joint_step=args.close_joint_step,
            target_rot=controller.home_rot,
            rot_gain=args.rot_gain,
        )

    controller.move_stage(
        [args.grasp_x, 0.0, args.grasp_z],
        steps=args.hold_steps,
        gripper=args.close_gripper,
        pos_gain=args.hold_gain,
        damp=args.close_damp,
        max_joint_step=args.close_joint_step,
        target_rot=controller.home_rot,
        rot_gain=args.rot_gain,
    )

    lift_positions = np.linspace(args.grasp_z, args.lift_z, args.lift_phases + 1)[1:]
    for z in lift_positions:
        controller.move_stage(
            [args.lift_x, 0.0, float(z)],
            steps=args.lift_steps,
            gripper=args.close_gripper,
            pos_gain=args.lift_gain,
            damp=args.lift_damp,
            max_joint_step=args.lift_joint_step,
            target_rot=controller.home_rot,
            rot_gain=args.rot_gain,
        )

    controller.move_stage(
        [args.lift_x, 0.0, args.lift_z],
        steps=args.settle_steps,
        gripper=args.close_gripper,
        pos_gain=args.lift_gain,
        damp=args.lift_damp,
        max_joint_step=args.lift_joint_step,
        target_rot=controller.home_rot,
        rot_gain=args.rot_gain,
    )
    return sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach-steps", type=int, default=500)
    parser.add_argument("--pregrasp-steps", type=int, default=500)
    parser.add_argument("--close-steps", type=int, default=500)
    parser.add_argument("--hold-steps", type=int, default=250)
    parser.add_argument("--lift-steps", type=int, default=260)
    parser.add_argument("--lift-phases", type=int, default=4)
    parser.add_argument("--settle-steps", type=int, default=250)
    parser.add_argument("--pregrasp-z", type=float, default=0.495)
    parser.add_argument("--grasp-z", type=float, default=0.472)
    parser.add_argument("--grasp-x", type=float, default=0.500)
    parser.add_argument("--lift-z", type=float, default=0.66)
    parser.add_argument("--lift-x", type=float, default=0.500)
    parser.add_argument("--open-gripper", type=float, default=255.0)
    parser.add_argument("--close-gripper", type=float, default=180.0)
    parser.add_argument("--close-gain", type=float, default=1.3)
    parser.add_argument("--hold-gain", type=float, default=1.0)
    parser.add_argument("--lift-gain", type=float, default=1.4)
    parser.add_argument("--close-damp", type=float, default=2e-2)
    parser.add_argument("--lift-damp", type=float, default=2e-2)
    parser.add_argument("--close-joint-step", type=float, default=0.03)
    parser.add_argument("--lift-joint-step", type=float, default=0.03)
    parser.add_argument("--rot-gain", type=float, default=0.5)
    parser.add_argument("--save", default="/work/final_state.npz")
    args = parser.parse_args()

    sim = build_trace(args)
    sim.save_final_state(args.save)
    result = replay_eval(np.array(sim._ctrl_trace))
    print(
        {
            "final_cup_z": round(result.final_cup_z, 4),
            "max_cup_z": round(result.max_cup_z, 4),
            "contact_fraction": round(result.contact_fraction, 4),
            "progress_score": round(result.progress_score, 4),
            "sim_time": round(sim.data.time, 4),
            "trace_len": len(sim._ctrl_trace),
            "cup_pos_end": np.round(sim.cup_position(), 4).tolist(),
            "grip_center_end": np.round(PandaCupController(sim).grip_center(), 4).tolist(),
        }
    )


if __name__ == "__main__":
    main()
