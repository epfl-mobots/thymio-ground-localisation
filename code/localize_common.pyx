#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal
# vim: ts=4:sw=4:noexpandtab

import numpy as np
from libc.math cimport floor
cimport numpy as np
cimport cython

# main class

cdef class AbstractLocalizer:

	# constructor
	def __init__(self, np.ndarray[double, ndim=2] ground_map):
	
		assert ground_map.dtype == np.double
		
		# copy parameters
		self.ground_map = ground_map
	
	# support methods
	
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