#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

import numpy as np
from libc.math cimport floor
cimport numpy as np
cimport cython

# support functions

cpdef np.ndarray[double, ndim=2] rot_mat2(double angle):
	""" Create a 2D rotation matrix for angle """
	return np.array([[np.cos(angle), -np.sin(angle)],
	                 [np.sin(angle),  np.cos(angle)]])

# main class

cdef class AbstractLocalizer:

	# constructor
	def __init__(self, np.ndarray[double, ndim=2] ground_map, double alpha_xy, double alpha_theta):

		assert ground_map.dtype == np.double

		# copy parameters
		self.ground_map = ground_map
		self.alpha_xy = alpha_xy
		self.alpha_theta = alpha_theta

	# support methods

	cpdef bint is_in_bound_cell(self, int x, int y):
		""" Return whether a given position x,y (as int) is within the bounds of a 2D array """
		if x >= 0 and y >= 0 and x < self.ground_map.shape[0] and y < self.ground_map.shape[1]:
			return True
		else:
			return False

	cpdef bint is_in_bound(self, double[:] pos):
		""" Check whether a given position is within the bounds of a 2D array """
		assert pos.shape[0] == 2
		cdef int x = self.xyW2C(pos[0])
		cdef int y = self.xyW2C(pos[1])
		return self.is_in_bound_cell(x,y)

	cpdef double xyC2W(self, int pos):
		""" Transform an x or y coordinate in cell coordinates into world coordinates """
		return pos+0.5

	cpdef int xyW2C(self, double pos):
		""" Transform an x or y coordinate in world coordinates into cell coordinates """
		return int(floor(pos))

	cpdef double dxyC2W(self, int dpos):
		""" Transform an x or y difference in cell coordinates into a difference in world coordinates """
		return float(dpos)

	cpdef int dxyW2C(self, double dpos):
		""" Transform an x or y difference in world coordinates into a difference in cell coordinates """
		return int(round(dpos))