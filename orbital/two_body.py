import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D

# plot background color
plt.style.use('dark_background')

earth_radius = 6378.0 #kilometers
earth_mu = 398600.0 # km^3/s^2


def plot(rs):
	fig = plt.figure(figsize=(18,6))
	ax = fig.add_subplot(111, projection = '3d')

	ax.plot(rs[:,0],rs[:,1], rs[:,2], 'k', label='Trajectory')
	#initial position in orbit
	ax.plot([rs[0,0]],[rs[0,1]],[rs[0,2]], 'wo', label = 'initial position')

	#plot central body
	_u,_v = np.mgrid[0:2*np.pi:20j,0:np.pi:10j]
	_x = earth_radius*np.cos(_u)*np.sin(_v)
	_y = earth_radius*np.sin(_u)*np.sin(_v)
	_z = earth_radius*np.cos(_v)
	
	# plot surface on 3d earth
	ax.plot_surface(_x,_y,_z, cmap='Blues')
	
	# plot the x,y,z vectors
	l = earth_radius * 2
	#arrow start x,y,z
	x,y,z = [[0,0,0],[0,0,0],[0,0,0]]
	#arrow end
	u,v,w = [[1,0,0],[0,1,0],[0,0,1]]
	#arrow animation library from matplotlib
	ax.quiver(x,y,z,u,v,w, color='k')
	
	max_val = np.max(np.abs(rs))
	
	ax.set_xlim([-max_val, max_val])
	ax.set_ylim([-max_val, max_val])
	ax.set_zlim([-max_val, max_val])
	
	ax.set_xlabel(['X (km)'])
	ax.set_ylabel(['Y (km)'])
	ax.set_zlabel(['Z (km)'])
	
	#ax.set_aspect('equal')
	
	#title and legend
	ax.set_title('Trajectory plot')
	plt.legend(['Trajectory', 'initial pos'])
	
	#plot
	plt.show()

#differential equation set up function 
def diffy_q(t,y,mu):
	#time (t), state(y), mu
	rx,ry,rz,vx,vy,vz = y
	r= np.array([rx,ry,rz])
	
	#normal vector of radius vector
	norm_r = np.linalg.norm(r)
	
	#two_body acceleration
	ax,ay,az = -r*mu/norm_r**3
	
	#return derivatives of the input(state = position, velocity)
	return[vx,vy,vz,ax,ay,az]
	

	

if __name__ == '__main__':
	#initial conditions of orbit parameters
	r_mag = earth_radius + 500.0 #km altitude
	v_mag = np.sqrt(earth_mu/r_mag) #km/s
	
	#initial position and velocity vectors (set up as a python list _-- concatenats later on)
	r0 = [r_mag, 0,0]
	#velocity perpindicular to position (for circular motion)
	v0 = [0,v_mag, 0]
	
	#timespan for running simulation
	tspan = 100*60.0
	
	#timestep
	dt = 100.0
	
	#total number of steps in simulation
	n_steps = int(np.ceil(tspan/dt))
	
	#initialize arrays
	ys = np.zeros((n_steps,6))
	ts = np.zeros((n_steps,1))
	
	#initial conditions
	y0 = r0+v0
	ts[0] = 0
	ys[0] = np.array(y0)
	step = 1
	
	
	#initiate solver for ode equations
	solver = ode(diffy_q)
	solver.set_integrator('lsoda')
	solver.set_initial_value(y0,0)
	solver.set_f_params(earth_mu)
	
	#propagate orbit
	while solver.successful() and step<n_steps:
		solver.integrate(solver.t+dt)
		ts[step] = solver.t
		ys[step] = solver.y
		step += 1
		
	rs = ys[:,:3]
	
plot(rs)


# add more satellites
