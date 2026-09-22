import numpy as np
import mujoco
from sim import Sim

p = np.array([
    8.97887350e-03, -7.99097214e-04, 6.80618308e-01,
   -1.03092180e-04,  1.17248641e-03, 4.68573571e-01,
    4.87935096e-03,  2.40690143e-02, 1.79560003e+02,
    1.04900000e+03, 6.14564775e-01, -7.70134733e-03,
    4.58030234e-03, 6.34193914e-01, 4.56000000e-01, 4.70000000e-01
])


def bid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


sim = Sim(); m=sim.model; d=sim.data
lf=bid(m,'left_finger'); rf=bid(m,'right_finger'); cup=bid(m,'cup')
ctrl_min=m.actuator_ctrlrange[:7,0]; ctrl_max=m.actuator_ctrlrange[:7,1]
qhome=np.array([0.0,0.3,0.0,-1.2,0.0,1.3,0.0])

def pinch(): return 0.5*(d.xpos[lf]+d.xpos[rf])
def cup_pos(): return np.array(d.xpos[cup])

def step_to(target, grip, steps, k=4.0, damp=1e-3, reg=0.04):
    for _ in range(steps):
        mujoco.mj_forward(m,d)
        e=target-pinch()
        Jl=np.zeros((3,m.nv)); Jr=np.zeros((3,m.nv)); z=np.zeros((3,m.nv))
        mujoco.mj_jacBodyCom(m,d,Jl,z,lf); mujoco.mj_jacBodyCom(m,d,Jr,z,rf)
        J=0.5*(Jl[:,:7]+Jr[:,:7])
        dq=np.linalg.solve(J.T@J+damp*np.eye(7), J.T@(k*e)-reg*(d.qpos[:7]-qhome))
        d.ctrl[:7]=np.clip(d.qpos[:7]+dq, ctrl_min, ctrl_max)
        d.ctrl[7]=grip
        sim.step(1)

c0=cup_pos().copy()
step_to(np.array([c0[0]+p[0], c0[1]+p[1], p[2]]), p[8], 700)
step_to(np.array([c0[0]+p[3], c0[1]+p[4], p[5]]), p[8], 700)
for _ in range(int(p[9])):
    d.ctrl[:7]=d.qpos[:7]; d.ctrl[7]=0.0; sim.step(1)
step_to(np.array([c0[0]+p[6], c0[1]+p[7], p[10]]), 0.0, 900)
step_to(np.array([0.70+p[11], 0.14+p[12], p[13]]), 0.0, 900)
step_to(np.array([0.70+p[11], 0.14+p[12], p[14]]), 0.0, 1100)
# very gentle release at low height
step_to(np.array([0.70+p[11], 0.14+p[12], p[15]]), 255.0, 1500)
# hold open mostly stationary so cup settles before retreat
step_to(np.array([0.70+p[11], 0.14+p[12], 0.50]), 255.0, 700)
step_to(np.array([0.58,0.16,0.62]), 255.0, 800)
sim.step(900)
sim.save_final_state('/work/final_state.npz')

# quick diagnostics
trace_contact=np.array([t['cup_contact'] for t in sim._trace],dtype=float)
bz=np.max(np.array([t['cup_pos'][2] for t in sim._trace]))
for _ in range(500): sim.step(1)
settle=[]
for _ in range(100):
    settle.append(sim.has_gripper_cup_contact()); sim.step(1)
upz=abs(d.xmat[cup].reshape(3,3)[2,2]); c=cup_pos()
print('saved', '/work/final_state.npz')
print('contact_any',bool(np.any(trace_contact>0.5)),'best_z',bz,'last10',float(np.mean(trace_contact[-10:])))
print('final',c,'inside',abs(c[0]-0.7)<=0.046 and abs(c[1]-0.14)<=0.046,'zok',c[2]>=0.42,'upz',upz,'settle',float(np.mean(settle)))
