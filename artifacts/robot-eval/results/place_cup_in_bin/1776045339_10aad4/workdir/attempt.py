from sim import Sim
import numpy as np, mujoco

sim=Sim()
# crude baseline: open, move to a reachable posture near the cup, close, lift, move toward bin, open.
qs = [
    np.array([0.0,0.0,0.0,-0.07,0.0,0.19,0.0]),
    np.array([-1.559, 1.114, 2.024, -2.076, -1.437, 0.958, -0.79]),
    np.array([-1.559, 1.114, 2.024, -2.076, -1.437, 1.4, -0.79]),
    np.array([-1.2, 0.9, 1.8, -1.8, -1.1, 1.2, -0.6]),
]
gs = [255,255,0,255]
for q,g in zip(qs,gs):
    sim.data.ctrl[:7]=q
    sim.data.ctrl[7]=g
    sim.step(300)
sim.step(200)
print('cup',sim.cup_position(),'contact',sim.has_gripper_cup_contact(),'time',sim.data.time)
sim.save_final_state('/work/final_state.npz')
print('saved')
