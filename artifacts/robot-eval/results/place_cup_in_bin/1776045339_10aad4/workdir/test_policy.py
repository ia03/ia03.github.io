from sim import Sim
import numpy as np, mujoco, os
from PIL import Image

def move(sim, q, g, steps=200):
    q0=sim.data.ctrl[:7].copy(); g0=sim.data.ctrl[7].copy()
    for i in range(steps):
        a=(i+1)/steps
        sim.data.ctrl[:7]=(1-a)*q0 + a*q
        sim.data.ctrl[7]=(1-a)*g0 + a*g
        sim.step(1)

def snap(sim, name):
    img=sim.render(640,480)
    Image.fromarray(img).save(name)

sim=Sim()
init=sim.data.qpos[:7].copy()
pre=np.array([-1.664, 1.466, 2.174, -2.489, -2.141, 1.291, -1.065])
grasp=np.array([-2.778, -0.742, 2.358, -2.109, -2.522, 0.132, -1.77 ])
lift=np.array([-2.323, -0.632, 1.991, -1.13 , 1.135, 1.72 , -2.765])
place=np.array([ 1.046,  1.194, -1.213, -1.522, -1.998,  1.368, -1.261])

sim.data.ctrl[:7]=init; sim.data.ctrl[7]=255
sim.step(100)
snap(sim,'/work/0_init.png')
move(sim, pre, 255, 400); snap(sim,'/work/1_pre.png')
move(sim, grasp, 255, 500); snap(sim,'/work/2_grasp_open.png')
move(sim, grasp, 0, 250); snap(sim,'/work/3_closed.png')
move(sim, lift, 0, 500); snap(sim,'/work/4_lift.png')
move(sim, place, 0, 600); snap(sim,'/work/5_place_hold.png')
move(sim, place, 255, 200); snap(sim,'/work/6_release.png')
sim.step(300); snap(sim,'/work/7_settle.png')
print('cup',sim.cup_position(),'contact',sim.has_gripper_cup_contact(),'time',sim.data.time)
print('trace max z', max(t['cup_pos'][2] for t in sim._trace), 'final contact frac last10', np.mean([t['cup_contact'] for t in sim._trace[-10:]]))
sim.save_final_state('/work/final_state.npz')
