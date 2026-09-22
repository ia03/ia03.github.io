import numpy as np
from sim import Sim, TOUCH_POINT, TARGET_XY

sim=Sim()
home=sim.home_ctrl()

# home hold
sim.data.ctrl[:] = home
sim.step(500)
print('home500 pelvis', sim.pelvis_position(), 'torso_up', sim.torso_up(), 'dtarget', sim.distance_to_target(), 'dtouch', sim.distance_to_touch())

# sinus walk attempt
sim.reset()
for t in range(2000):
    u=home.copy()
    ph=2*np.pi*t/120
    # leg swing
    a=0.22
    u[2] += a*np.sin(ph)   # l hip pitch
    u[7] += -a*np.sin(ph)  # r hip pitch
    u[3] += -0.35*a*np.sin(ph)
    u[8] += 0.35*a*np.sin(ph)
    # slight torso lean forward
    u[10]=0.10
    sim.data.ctrl[:]=u
    sim.step(1)
print('walk2000 pelvis', sim.pelvis_position(), 'torso_up', sim.torso_up(), 'dtarget', sim.distance_to_target(), 'dtouch', sim.distance_to_touch())
