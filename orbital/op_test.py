import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

import os 
from orbit_propagator import orbit_propagator as OP
import planetary_data as pd

cb = pd.earth

if __name__ == '__main__':
	#initial magnitude of orbit params
	r_mag = cb['radius'] + 1500.0
	v_mag = np.sqrt(cb['mu']/r_mag)
	
	#initial position and velocity vectors
	r0 = [r_mag,r_mag*0.01,r_mag*-0.1]
	v0 = [0,v_mag,v_mag*0.3]
	
	# 1 day 
	tspan = 3600*24.0
	
	# 100 seconds 
	dt = 100.0
	
	op = OP(r0,v0,tspan, dt)
	op.propagate_orbit()
	op.plot_3d(show_plot=True)