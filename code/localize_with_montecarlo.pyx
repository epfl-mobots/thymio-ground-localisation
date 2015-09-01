# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

import pyximport; pyximport.install()
import numpy as np
import math
import bisect
import random
from libc.math cimport floor, sqrt, log, atan2, sin, cos, exp
cimport numpy as np
cimport cython
import localize_common
cimport localize_common
from localize_common import rot_mat2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# some useful constants in local scope

cdef double _pi = math.pi
cdef double _1sqrt2pi = 1. / sqrt(2. * math.pi)

@cython.cdivision(True) # turn off division-by-zero checking
cdef double _norm(double x, double u, double s):
	cdef double factor = _1sqrt2pi / s
	cdef double dxus = (x - u) / s
	return factor * exp(- (dxus * dxus) / 2.)

# main class

cdef class MCLocalizer(localize_common.AbstractLocalizer):

	# user parameters
	cdef int N_uniform

	# particles
	cdef double[:,:] particles # 2D array of particles_count x (x,y,theta)

	def __init__(self, np.ndarray[double, ndim=2] ground_map, int particles_count, double sigma_obs, double prob_uniform, double alpha_xy, double alpha_theta):
		""" Create the localizer with the ground map and some parameters """

		super(MCLocalizer, self).__init__(ground_map, alpha_xy, alpha_theta, sigma_obs)

		# setup parameters
		self.N_uniform = int(prob_uniform*particles_count)

		# create initial particles filled the whole space
		cdef np.ndarray[double, ndim=2] particles = np.random.uniform(0,1,[particles_count, 3])
		particles *= [ground_map.shape[0], ground_map.shape[1], _pi*2]
		self.particles = particles

	def apply_obs(self, double left_color, double right_color):
		""" Apply observation and resample """

		cdef int i
		cdef double x, y, theta
		cdef np.ndarray[double, ndim=2] R
		cdef double left_weight, right_weight
		cdef int x_i, y_i, particle_index
		cdef double ground_val
		cdef double sigma = self.sigma_obs
		cdef int particles_count = self.particles.shape[0]
		cdef int resample_count = particles_count - self.N_uniform
		cdef int uniform_count = self.N_uniform
		cdef np.ndarray[double, ndim=1] weights = np.empty([particles_count])

		# matching particles
		nb_ok = 0
		# apply observation to every particle
		for i, (x, y, theta) in enumerate(self.particles):

			# compute position of sensors in world coordinates
			R = rot_mat2(theta)
			left_sensor_pos = R.dot([7.2, 1.1]) + [x,y]
			right_sensor_pos = R.dot([7.2, -1.1]) + [x,y]

			if not self.is_in_bound(left_sensor_pos) or not self.is_in_bound(right_sensor_pos):
				# kill particle if out of map
				weights[i] = 0.
			else:
				# otherwise, compute weight in function of ground color

				# left sensor
				ground_val = self.ground_map[self.xyW2C(left_sensor_pos[0]), self.xyW2C(left_sensor_pos[1])]
				left_weight = _norm(left_color, ground_val, sigma)

				# right sensor
				ground_val = self.ground_map[self.xyW2C(right_sensor_pos[0]), self.xyW2C(right_sensor_pos[1])]
				right_weight = _norm(right_color, ground_val, sigma)

				# compute weight
				weights[i] = left_weight * right_weight

			# update matching particles
			if weights[i] > 0.5:
				nb_ok += 1

		# ratio matching particles
		print "  Proportion of matching particles:", 1.*nb_ok/len(weights)

		# resample
		assert weights.sum() > 0.
		weights /= weights.sum()
		cdef np.ndarray[double, ndim=2] particles_view = np.asarray(self.particles)
		cdef np.ndarray[double, ndim=2] resampled = particles_view[np.random.choice(particles_count, resample_count, p=weights)]
		cdef np.ndarray[double, ndim=2] new_particles = np.random.uniform(0,1,[uniform_count, 3]) * [self.ground_map.shape[0], self.ground_map.shape[1], _pi*2]
		# FIXME I don't know why that doesn't work so I copy manually -_-
		#self.particles[:resample_count] = resampled
		#self.particles[resample_count:] = new_particles
		for i, p in enumerate(resampled):
			for j, v in enumerate(p):
				self.particles[i,j] = v
		for i, p in enumerate(new_particles):
			for j, v in enumerate(p):
				self.particles[i+resample_count,j] = v
		# end FIXME


	def apply_command(self, d_x, d_y, d_theta):
		""" Apply command to each particle """

		cdef int i
		cdef double x, y, theta

		# error model, same as with CPT, but without added half cell
		cdef double norm_xy = sqrt(d_x*d_x + d_y*d_y)
		cdef double e_theta = self.alpha_theta * math.fabs(d_theta) + math.radians(0.25)
		assert e_theta > 0, e_theta
		cdef double e_xy = self.alpha_xy * norm_xy + 0.01
		assert e_xy > 0, e_xy

		# apply command and sampled noise to each particle
		for i, (x, y, theta) in enumerate(self.particles):
			np.asarray(self.particles)[i,0:2] += rot_mat2(theta).dot([d_x, d_y]) + np.random.normal(0, e_xy, [2])
			self.particles[i,2] += d_theta + np.random.normal(0, e_theta)

	def estimate_state(self):
		# return the mean of the distribution
		sins = [sin(theta) for theta in self.particles[:,2]]
		coss = [cos(theta) for theta in self.particles[:,2]]
		s_sins = sum(sins)
		s_coss = sum(coss)
		theta_m = atan2(s_sins, s_coss)
		x_m, y_m = np.mean(self.particles[:,0:2], 0)
		return np.array([x_m, y_m, theta_m])
		## return a random particle
		#cdef int i = random.randint(0, self.particles.shape[0]-1)
		#return np.array(self.particles[i])

	def estimate_logratio(self, double x, double y, double theta):
		# TODO
		return 0

	# debug methods

	def dump_PX(self, str base_filename, float gt_x = -1, float gt_y = -1, float gt_theta = -1):
		""" Write particles to an image """
		fig = Figure((3,3), tight_layout=True)
		canvas = FigureCanvas(fig)
		ax = fig.gca()
		ax.set_xlim([0, self.ground_map.shape[0]])
		ax.set_ylim([0, self.ground_map.shape[1]])

		for (x, y, theta) in self.particles:
			ax.arrow(x, y, math.cos(theta), math.sin(theta), head_width=0.8, head_length=1, fc='k', ec='k', alpha=0.3)

		ax.arrow(gt_x, gt_y, math.cos(gt_theta)*2, math.sin(gt_theta)*2, head_width=1, head_length=1.2, fc='blue', ec='blue')

		canvas.print_figure(base_filename+'.png', dpi=300)
