import numpy as np
from math import sqrt
import os
import planetary_data as pd
import tools as t
from orbit_propagator import orbit_propagator as OP

#time parameters
tspan = 3600*24*1.0
dt = 100.0

if __name__ == '__main__':

	r_mag = pd.earth['radius'] + 400.0
	v_mag = sqrt(pd.earth['mu']/r_mag)
	r0 = np.array([r_mag, 0,0])
	v0 = np.array([0,v_mag,0])
	
	r_mag = pd.earth['radius'] + 1000.0
	v_mag = sqrt(pd.earth['mu']/r_mag)*1.3
	r00 = np.array([r_mag, 0,0])
	v00 = np.array([0,v_mag,0.3])
	
	op0 = OP(r0,v0,tspan,dt)
	op00 = OP(r00,v00, tspan, dt)
	
	op0.propagate_orbit()
	op00.propagate_orbit()
	
	t.plot_n_orbits([op0.rs, op00.rs], labels = ['0','1'], show_plot=True)