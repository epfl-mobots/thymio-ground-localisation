# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal
# vim: ts=4:sw=4:noexpandtab

import numpy as np
cimport numpy as np
cimport cython

# main class

cdef class AbstractLocalizer:

	# map
	cdef double[:,:] ground_map
	
	# support methods
	cpdef double xyC2W(self, int pos)
	cpdef int xyW2C(self, double pos)
	cpdef double dxyC2W(self, int dpos)
	cpdef int dxyW2C(self, double dpos)