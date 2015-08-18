# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal
# vim: ts=4:sw=4:noexpandtab

import numpy as np
import math
import scipy.misc
from libc.math cimport floor
cimport numpy as np
cimport cython

# some useful constants in local scope

cdef double _pi = math.pi
cdef double _1pi = 1. / math.pi

# support functions

cdef double _angle_steps_to_angle(int angle, int angle_N):
	""" Transform an angle in steps to a real angle """
	return ((angle+0.5) * 2 * _pi) / angle_N
	
cdef np.ndarray[double, ndim=2] _rot_mat2(double angle):
	""" Create a 2D rotation matrix for angle """
	return np.array([[np.cos(angle), -np.sin(angle)],
	                 [np.sin(angle),  np.cos(angle)]])

cdef bint _is_in_bound_int(int x, int y, int w, int h):
	""" Return whether a given position x,y (as int) is within the bounds of a 2D array """
	if x >= 0 and y >= 0 and x < w and y < h:
		return True
	else:
		return False

cdef bint _is_in_bound(double[:] pos, int w, int h):
	""" Check whether a given position is within the bounds of a 2D array """
	assert pos.shape[0] == 2
	cdef int x = int(floor(pos[0]))
	cdef int y = int(floor(pos[1]))
	return _is_in_bound_int(x,y,w,h)

# main class

cdef class CPTLocalizer:

	# parameters
	cdef int angle_N
	
	# observation model structures
	cdef double[:,:,:] obs_left_black
	cdef double[:,:,:] obs_left_white
	cdef double[:,:,:] obs_right_black
	cdef double[:,:,:] obs_right_white
	
	# probability distribution for latent space
	cdef double[:,:,:] PX
	
	# constructor
	def __init__(self, np.ndarray[double, ndim=2] ground_map, int angle_N, double prob_correct):
		""" Fill the tables obs_left/right_black/white of the same resolution as the ground_map and an angle discretization angle_N """
		
		assert ground_map.dtype == np.double
		
		# copy parameters
		self.angle_N = angle_N
		
		# create the arrays
		cdef shape = [angle_N, ground_map.shape[0], ground_map.shape[1]]
		self.obs_left_black = np.empty(shape, np.double)
		self.obs_left_white = np.empty(shape, np.double)
		self.obs_right_black = np.empty(shape, np.double)
		self.obs_right_white = np.empty(shape, np.double)
		
		# pre-compute the shift vectors to lookup the map from a given robot position
		cdef np.ndarray[double, ndim=2] shifts_left = np.empty([angle_N, 2], np.double)
		cdef np.ndarray[double, ndim=2] shifts_right = np.empty([angle_N, 2], np.double)
		cdef np.ndarray[double, ndim=2] R
		cdef double theta
		cdef int i, j, k
		for i in range(angle_N):
			theta = _angle_steps_to_angle(i, angle_N)
			R = _rot_mat2(theta)
			shifts_left[i,:] = R.dot([7.2, 1.1])
			shifts_right[i,:] = R.dot([7.2, -1.1])
		
		# DEBUG
		#print 'shifts_left = ', shifts_left
		#print 'shifts_right = ', shifts_right
		
		# fill the cells
		cdef int x, y, w, h
		w, h = ground_map.shape[0], ground_map.shape[1]
		cdef double prob_wrong = 1.0 - prob_correct
		for i in range(angle_N):
			for j in range(ground_map.shape[0]):
				for k in range(ground_map.shape[1]):
					cell_center = [j+0.5, k+0.5]
					# left sensor
					x,y = (shifts_left[i,:] + cell_center).astype(int)
					if _is_in_bound_int(x,y,w,h):
						if ground_map[x,y] == 1.:
							self.obs_left_black[i,j,k] = prob_correct
							self.obs_left_white[i,j,k] = prob_wrong
						else:
							self.obs_left_black[i,j,k] = prob_wrong
							self.obs_left_white[i,j,k] = prob_correct
					else:
						self.obs_left_black[i,j,k] = 0.5
						self.obs_left_white[i,j,k] = 0.5
					# right sensor
					x,y = (shifts_right[i,:] + cell_center).astype(int)
					if _is_in_bound_int(x,y,w,h):
						if ground_map[x,y] == 1.:
							self.obs_right_black[i,j,k] = prob_correct
							self.obs_right_white[i,j,k] = prob_wrong
						else:
							self.obs_right_black[i,j,k] = prob_wrong
							self.obs_right_white[i,j,k] = prob_correct
					else:
						self.obs_right_black[i,j,k] = 0.5
						self.obs_right_white[i,j,k] = 0.5
		
		# initialize PX
		self.PX = np.empty(shape, np.double)
		np.asarray(self.PX).fill(0.5)
	
	# main methods
	
	def apply_obs(self, bint is_left_black, bint is_right_black):
		""" Update the latent space with observation """
		
		# update PX
		if is_left_black:
			self.PX = np.asarray(self.PX) * self.obs_left_black
		else:
			self.PX = np.asarray(self.PX) * self.obs_left_white
		if is_right_black:
			self.PX = np.asarray(self.PX) * self.obs_right_black
		else:
			self.PX = np.asarray(self.PX) * self.obs_right_white
		
		# renormalize PX
		self.PX /= self.PX.sum()
		# FIXME: time it and maybe do not do it always
	
	def apply_command(self, double d_undef0, double d_undef1, double d_theta):
		""" TODO: document once we know the semantics """
		
		# first, compute Gaussian errors
		cdef double[:,:,:] sigmas = np.zeros([self.angle_N, 3, 3], np.double)
		cdef double theta_dest
		cdef int i
		for i in range(self.angle_N):
			theta_dest = _angle_steps_to_angle(i, self.angle_N) + d_theta
			# TODO: continue here
			pass
	
	# debug methods
	
	def dump_obs_model(self, str base_filename):
		""" Write images of observation model """
		cdef int i
		for i in range(self.angle_N):
			scipy.misc.imsave(base_filename+'-'+str(i)+'-left_black.png', self.obs_left_black[i])
			scipy.misc.imsave(base_filename+'-'+str(i)+'-left_white.png', self.obs_left_white[i])
			scipy.misc.imsave(base_filename+'-'+str(i)+'-right_black.png', self.obs_right_black[i])
			scipy.misc.imsave(base_filename+'-'+str(i)+'-right_white.png', self.obs_right_white[i])
