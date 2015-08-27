# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal
# vim: ts=4:sw=4:noexpandtab

import pyximport; pyximport.install()
import numpy as np
import math
import bisect
import random
from libc.math cimport floor, sqrt, log
cimport numpy as np
cimport cython
import localize_common
cimport localize_common
from localize_common import rot_mat2

# some useful constants in local scope

cdef double _pi = math.pi

# helper functions

# from http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python
class WeightedRandomGenerator(object):
	def __init__(self, weights):
		self.totals = []
		running_total = 0
		
		for w in weights:
			running_total += w
			self.totals.append(running_total)

	def next(self):
		rnd = random.random() * self.totals[-1]
		return bisect.bisect_right(self.totals, rnd)
	
	def __iter__(self):
		return self

# main class

cdef class MCLocalizer(localize_common.AbstractLocalizer):

	# user parameters
	cdef double prob_correct
	
	# fixed/computed parameters
	cdef double alpha_theta_to_xy
	cdef double alpha_xy_to_xy
	cdef double alpha_theta_to_theta
	cdef double alpha_xy_to_theta
	
	# particles
	cdef double[:,:] particles # 2D array of particles_count x (x,y,theta)
	
	def __init__(self, np.ndarray[double, ndim=2] ground_map, int particles_count, double prob_correct):
		""" Create the localizer with the ground map and some parameters """
		
		super(MCLocalizer, self).__init__(ground_map)
		
		# setup parameters
		self.prob_correct = prob_correct
		self.alpha_theta_to_xy = 0.1
		self.alpha_xy_to_xy = 0.1
		self.alpha_theta_to_theta = 0.1
		self.alpha_xy_to_theta = 0.05
		
		# create initial particles filled the whole space
		cdef np.ndarray[double, ndim=2] particles = np.random.uniform(0,1,[particles_count, 3])
		particles *= [ground_map.shape[0], ground_map.shape[1], _pi*2]
		self.particles = particles
		
	def apply_obs(self, is_left_black, is_right_black):
		""" Apply observation and resample """
		
		cdef int i
		cdef double x, y, theta
		cdef np.ndarray[double, ndim=2] R
		cdef double left_weight, right_weight
		cdef int x_i, y_i, particle_index
		cdef int particles_count = self.particles.shape[0]
		cdef np.ndarray[double, ndim=1] weights = np.empty([particles_count])
		
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
				# WARNING: value to color encoding is unusual:
				# black is 1, white is 0
				# left sensor
				
				if self.ground_map[self.xyW2C(left_sensor_pos[0]), self.xyW2C(left_sensor_pos[1])] == 1 and is_left_black:
					left_weight = self.prob_correct
				else:
					left_weight = 1.0 - self.prob_correct
				# right sensor
				
				if self.ground_map[self.xyW2C(right_sensor_pos[0]), self.xyW2C(right_sensor_pos[1])] == 1 and is_right_black:
					right_weight = self.prob_correct
				else:
					right_weight = 1.0 - self.prob_correct
				# compute weight
				weights[i] = left_weight * right_weight
		
		# resample
		assert weights.sum() > 0.
		cdef np.ndarray[double, ndim=2] new_particles = np.empty(np.asarray(self.particles).shape)
		for i, particle_index in zip(range(particles_count), WeightedRandomGenerator(weights)):
			#print i, particle_index
			new_particles[i] = self.particles[particle_index]
		self.particles = new_particles
	
	def apply_command(self, d_x, d_y, d_theta):
		""" Apply command to each particle """
		
		cdef int i
		cdef double x, y, theta
		
		# error model, same as with CPT, but without added half cell
		cdef double norm_xy = sqrt(d_x*d_x + d_y*d_y)
		cdef double e_theta = self.alpha_xy_to_theta * norm_xy + self.alpha_theta_to_theta * d_theta
		cdef double e_xy = self.alpha_xy_to_xy * norm_xy + self.alpha_theta_to_xy * d_theta 
		
		# apply command and sampled noise to each particle
		for i, (x, y, theta) in enumerate(self.particles):
			np.asarray(self.particles)[i,0:2] += rot_mat2(theta).dot([d_x, d_y]) + np.random.normal(0, e_xy, [2])
			self.particles[i,2] += d_theta + np.random.normal(0, e_theta)

	def estimate_state(self):
		# return a random particle
		cdef int i = random.randint(0, self.particles.shape[0]-1)
		return np.array(self.particles[i])
		
	def estimate_logratio(self, double x, double y, double theta):
		# TODO
		return 0
		