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
	cdef double eps_x
	cdef double eps_y
	cdef double eps_theta
	cdef double half_theta_step
	
	# map
	cdef double[:,:] ground_map
	
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
		self.eps_x = 1.
		self.eps_y = 1.
		self.eps_theta = 0.2087
		self.half_theta_step = math.radians(180. * angle_N)
		self.ground_map = ground_map
		
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
			theta = self._thetaC2W(i)
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
		cdef double c_x, c_y
		for i in range(angle_N):
			for j in range(ground_map.shape[0]):
				for k in range(ground_map.shape[1]):
					c_x = self._xC2W(j)
					c_y = self._yC2W(k)
					# left sensor
					x = self._xW2C(shifts_left[i,0] + c_x)
					y = self._yW2C(shifts_left[i,1] + c_y)
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
					x = self._xW2C(shifts_right[i,0] + c_x)
					y = self._yW2C(shifts_right[i,1] + c_y)
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
	
	def apply_command(self, double d_t, double d_x, double d_y, double d_theta):
		""" TODO: document once we are sure of the semantics """
		
		# first, compute Gaussian errors for different angles
		cdef np.ndarray[double, ndim=3] sigmas_inverse = np.zeros([self.angle_N, 3, 3], np.double)
		# errors
		cdef double e_x, e_y, e_theta    # errors
		cdef double s_theta, s_x, s_y    # source pos in world coordinates
		cdef double t_theta, t_x, t_y    # target pos in world coordinates
		cdef int t_theta_i, t_x_y, t_y_i # target pos in cell coordinates
		cdef np.ndarray[double, ndim=2] T, sigma
		cdef int i, j, k
		for i in range(self.angle_N):
			s_theta = self._thetaC2W(i)
			t_theta = s_theta + d_theta
			T = _rot_mat2(t_theta)
			# compute the error and add half a cell
			e_x = self.eps_x * d_t + 0.5
			e_y = self.eps_y * d_t + 0.5
			e_theta = self.eps_theta + self.half_theta_step
			sigma = np.zeros([3,3], np.double)
			sigma[0:2,0:2] = T.dot([[e_x, 0],[0, e_y]]).dot(np.transpose(T))
			sigma[2,2] = e_theta
			sigmas_inverse[i] = np.linalg.inv(sigma)
			print ''
			print e_x, e_y
			print T
			print T.dot([[e_x, 0],[0, e_y]])
			print np.transpose(T)
			print T.dot([[e_x, 0],[0, e_y]]).dot(np.transpose(T))
			print sigma
			print sigmas_inverse[i]
			# TODO: multiplication by transposition returns identity, why do we do that in the first place?
			
			t_theta_i = self._thetaW2C(t_theta)
			for j in range(self.ground_map.shape[0]):
				s_x = self._xC2W(j)
				t_x = s_x + d_x
				t_x_i = self._xW2C(t_x)
				for k in range(self.ground_map.shape[1]):
					s_y = self._yC2W(k)
					t_y = s_y + d_y
					t_y_i = self._yW2C(t_y)
					pass
			
					# TODO: continue implementing
			
	# debug methods
	
	def dump_obs_model(self, str base_filename):
		""" Write images of observation model """
		cdef int i
		for i in range(self.angle_N):
			scipy.misc.imsave(base_filename+'-'+str(i)+'-left_black.png', self.obs_left_black[i])
			scipy.misc.imsave(base_filename+'-'+str(i)+'-left_white.png', self.obs_left_white[i])
			scipy.misc.imsave(base_filename+'-'+str(i)+'-right_black.png', self.obs_right_black[i])
			scipy.misc.imsave(base_filename+'-'+str(i)+'-right_white.png', self.obs_right_white[i])
	
	# support functions
	
	cdef double _thetaC2W(self, int angle):
		""" Transform an angle in cell coordinates into an angle in world coordinates """
		return ((angle+0.5) * 2. * _pi) / self.angle_N

	cdef int _thetaW2C(self, double angle):
		""" Transform an angle in world coordinates into an angle in cell coordinates """
		return int(floor((angle * self.angle_N) / (2. * _pi)))
	
	cdef double _xC2W(self, int i):
		""" Transform an x coordinate in cell coordinates into world coordinates """
		return i+0.5
		
	cdef int _xW2C(self, double x):
		""" Transform an x coordinate in world coordinates into cell coordinates """
		return int(floor(x))
	
	cdef double _yC2W(self, int i):
		""" Transform an y coordinate in cell coordinates into world coordinates """
		return i+0.5
		
	cdef int _yW2C(self, double y):
		""" Transform an y coordinate in world coordinates into cell coordinates """
		return int(floor(y))
