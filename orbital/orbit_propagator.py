import os 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import planetary_data as pd
import tools as t
cb = pd.earth

class orbit_propagator:
	def __init__(self, r0, v0, tspan, dt, cb=pd.earth):
		self.r0=r0
		self.v0=v0
		self.y0 = self.r0.tolist() + self.v0.tolist()
		self.tspan=tspan
		self.dt=dt
		self.cb=cb
		
		#total number of steps in simulation
		self.n_steps = int(np.ceil(self.tspan/self.dt))
	
		#initialize arrays
		self.ts = np.zeros((self.n_steps,1))
		self.ys = np.zeros((self.n_steps,6))
		
	
		#initial conditions
		#self.y0 = self.r0 + self.v0
		self.ts[0]=0
		self.ys[0,:] = self.y0
		self.step = 1
	
	
		#initiate solver for ode equations
		self.solver = ode(self.diffy_q)
		self.solver.set_integrator('lsoda')
		self.solver.set_initial_value(self.y0,0)
		#solver.set_f_params(earth_mu)
		
	def propagate_orbit(self):
	
		#propagate orbit
		while self.solver.successful() and self.step<self.n_steps:
			self.solver.integrate(self.solver.t+self.dt)
			self.ts[self.step] = self.solver.t
			self.ys[self.step] = self.solver.y
			self.step += 1
		
		self.rs = self.ys[:,:3]
		self.vs = self.ys[:,3:]
		
	def diffy_q(self,t,y):
		#time (t), state(y), mu
		rx,ry,rz,vx,vy,vz = y
		r= np.array([rx,ry,rz])
	
		#normal vector of radius vector
		norm_r = np.linalg.norm(r)
	
		#two_body acceleration
		ax,ay,az = -r*self.cb['mu']/norm_r**3
	
		#return derivatives of the input(state = position, velocity)
		return[vx,vy,vz,ax,ay,az]

	def plot_3d(self, show_plot = False, save_plot=False, title='Test Title'):
		fig = plt.figure(figsize=(16,8))
		ax = fig.add_subplot(111, projection = '3d')

		ax.plot(self.rs[:,0],self.rs[:,1], self.rs[:,2], 'k', label='Trajectory')
		#initial position in orbit
		ax.plot([self.rs[0,0]],[self.rs[0,1]],[self.rs[0,2]], 'wo', label = 'initial position')

		#plot central body
		_u,_v = np.mgrid[0:2*np.pi:20j,0:np.pi:10j]
		_x = self.cb['radius']*np.cos(_u)*np.sin(_v)
		_y = self.cb['radius']*np.sin(_u)*np.sin(_v)
		_z = self.cb['radius']*np.cos(_v)
	
		# plot surface on 3d earth
		ax.plot_surface(_x,_y,_z, cmap='Blues')
	
		# plot the x,y,z vectors
		l = cb['radius'] * 2
		#arrow start x,y,z
		x,y,z = [[0,0,0],[0,0,0],[0,0,0]]
		#arrow end
		u,v,w = [[1,0,0],[0,1,0],[0,0,1]]
		#arrow animation library from matplotlib
		ax.quiver(x,y,z,u,v,w, color='k')
	
		max_val = np.max(np.abs(self.rs))
	
		ax.set_xlim([-max_val, max_val])
		ax.set_ylim([-max_val, max_val])
		ax.set_zlim([-max_val, max_val])
		
		ax.set_xlabel(['X (km)'])
		ax.set_ylabel(['Y (km)'])
		ax.set_zlabel(['Z (km)'])
	
		#ax.set_aspect('equal')
	
		#title and legend
		ax.set_title(title)
		plt.legend(['Trajectory', 'initial pos'])
	
		#plot
		#plt.show()
		
		if show_plot:
			plt.show()
		if save_plot:
			plt.savefig(title+'.png', dpi =300)