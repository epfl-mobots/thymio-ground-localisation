# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal
# vim: ts=4:sw=4:noexpandtab

import pyximport; pyximport.install()
import numpy as np
import math
import scipy.misc
from scipy.stats import norm
from libc.math cimport floor, sqrt, log
cimport numpy as np
cimport cython
import localize_common
cimport localize_common
from localize_common import rot_mat2

# some useful constants in local scope

cdef double _pi = math.pi
cdef double _1pi = 1. / math.pi

# support functions

# taken from http://stackoverflow.com/questions/11615664/multivariate-normal-density-in-python
def _norm_pdf_multivariate(x, mu, sigma):
	""" multivariate PDF for Gaussian """
	size = len(x)
	if size == len(mu) and (size, size) == sigma.shape:
		det = np.linalg.det(sigma)
		if det == 0:
			raise NameError("The covariance matrix can't be singular")

		norm_const = 1.0/ ( math.pow((2*_pi),float(size)/2) * math.pow(det,1.0/2) )
		x_mu = np.matrix(x - mu)
		inv = np.linalg.inv(sigma)
		try:
			result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
		except ArithmeticError as e:
			print 'Error computing Gaussian with following parameters'
			print 'x_mu', x_mu
			print 'inv_sigma', inv
			raise e
		return norm_const * result
	else:
		raise NameError("The dimensions of the input don't match")

# main class

cdef class CPTLocalizer(localize_common.AbstractLocalizer):

	# user parameters
	cdef int angle_N
	cdef double max_prob_error
	
	# fixed/computed parameters
	cdef double alpha_theta_to_xy
	cdef double alpha_xy_to_xy
	cdef double alpha_theta_to_theta
	cdef double alpha_xy_to_theta
	
	# observation model structures
	cdef double[:,:,:] obs_left_black
	cdef double[:,:,:] obs_left_white
	cdef double[:,:,:] obs_right_black
	cdef double[:,:,:] obs_right_white
	
	# probability distribution for latent space
	cdef double[:,:,:] PX
	
	# constructor
	
	def __init__(self, np.ndarray[double, ndim=2] ground_map, int angle_N, double prob_correct, double max_prob_error):
		""" Fill the tables obs_left/right_black/white of the same resolution as the ground_map and an angle discretization angle_N """
		
		super(CPTLocalizer, self).__init__(ground_map)
		
		# copy parameters
		assert angle_N != 0
		self.angle_N = angle_N
		self.max_prob_error = max_prob_error
		self.alpha_theta_to_xy = 0.1
		self.alpha_xy_to_xy = 0.1
		self.alpha_theta_to_theta = 0.1
		self.alpha_xy_to_theta = 0.05
		
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
			theta = self.thetaC2W(i)
			R = rot_mat2(theta)
			shifts_left[i,:] = R.dot([7.2, 1.1])
			shifts_right[i,:] = R.dot([7.2, -1.1])
		
		# fill the cells
		cdef int x, y
		cdef double prob_wrong = 1.0 - prob_correct
		cdef double c_x, c_y
		for i in range(angle_N):
			for j in range(ground_map.shape[0]):
				for k in range(ground_map.shape[1]):
					c_x = self.xyC2W(j)
					c_y = self.xyC2W(k)
					# WARNING: value to color encoding is unusual:
					# black is 1, white is 0
					# left sensor
					x = self.xyW2C(shifts_left[i,0] + c_x)
					y = self.xyW2C(shifts_left[i,1] + c_y)
					if self.is_in_bound_cell(x,y):
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
					x = self.xyW2C(shifts_right[i,0] + c_x)
					y = self.xyW2C(shifts_right[i,1] + c_y)
					if self.is_in_bound_cell(x,y):
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
		self.PX = np.ones(shape, np.double) / float(np.prod(shape))
	
	
	# main methods
	
	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.cdivision(True) # turn off division-by-zero checking
	def apply_obs(self, bint is_left_black, bint is_right_black):
		""" Update the latent space with observation """
		
		# create a view on the array to perform numpy operations such as *= or /=
		cdef np.ndarray[double, ndim=3] PX_view = np.asarray(self.PX)
		
		# update PX
		if is_left_black:
			PX_view *= self.obs_left_black
		else:
			PX_view *= self.obs_left_white
		if is_right_black:
			PX_view *= self.obs_right_black
		else:
			PX_view *= self.obs_right_white
		
		# renormalize PX
		PX_view /= PX_view.sum()
		# FIXME: time it and maybe do not do it always
	
	
	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.cdivision(True) # turn off division-by-zero checking
	@cython.wraparound(False) # turn off wrap-around checking
	@cython.nonecheck(False) # turn off 
	def apply_command(self, double d_x, double d_y, double d_theta):
		""" Apply a command for a displacement of d_x,d_y (in local frame) and a rotation of d_theta """
		
		# variables 
		cdef int i, j, k                 # outer loops indices
		cdef int d_i, d_j, d_k           # inner loops indices
		cdef double s_theta, s_x, s_y    # source pos in world coordinates
		cdef double t_theta, t_x, t_y    # center target pos in world coordinates (mean)
		cdef int u_theta_i, u_x_i, u_y_i # current target pos in cell coordinates
		cdef double d_x_r, d_y_r         # rotated d_x and d_y in function of theta in world coordinates
		cdef int d_theta_i, d_x_r_i, d_y_r_i     # rotated d_x and d_y in function of theta in cell coordinates
		cdef double d_theta_d, d_x_r_d, d_y_r_d  # diff between d_x/y_r and d_x/y_r_i
		cdef np.ndarray[double, ndim=2] T, sigma # cov. motion model
		
		# assertion and copy some values for optimisation
		assert self.ground_map is not None
		assert self.PX is not None
		assert self.ground_map.shape[0] == self.PX.shape[1]
		assert self.ground_map.shape[1] == self.PX.shape[2]
		cdef int w = self.ground_map.shape[0]
		cdef int h = self.ground_map.shape[1]
		cdef int angle_N = self.angle_N
		
		# error model for motion, inspired from 
		# http://www.mrpt.org/tutorials/programming/odometry-and-motion-models/probabilistic_motion_models/
		# sum of factors from translation (x,y), rotation (theta), and half a cell (for sampling issues)
		cdef double norm_xy = sqrt(d_x*d_x + d_y*d_y)
		cdef double e_theta = self.alpha_xy_to_theta * norm_xy + self.alpha_theta_to_theta * math.fabs(d_theta) + self.dthetaC2W(1) / 2.
		assert e_theta > 0, e_theta
		cdef double e_xy = self.alpha_xy_to_xy * norm_xy + self.alpha_theta_to_xy * math.fabs(d_theta) + self.dxyC2W(1) / 2.
		assert e_xy > 0, e_xy
		cdef np.ndarray[double, ndim=2] e_xy_mat = np.array([[e_xy, 0], [0, e_xy]])
		
		# special case if e_xy is huge, robot is most likely lost
		
		# compute how many steps around we have to compute to have less than 1 % of error in transfering probability mass
		# and allocate arrays for fast lookup
		# for theta
		cdef object e_theta_dist = norm(0, e_theta)
		cdef double e_theta_max = e_theta_dist.pdf(0)
		cdef double e_i
		#print 'e_theta', e_theta
		#print 'e_theta_max', e_theta_max 
		for i in range(1, angle_N/2):
			e_i = e_theta_dist.pdf(self.dthetaC2W(i))
			if e_i < self.max_prob_error * e_theta_max:
				break
		cdef int d_theta_range = i
		cdef int d_theta_shape = i*2 + 1
		cdef np.ndarray[double, ndim=1] e_theta_p = np.empty(d_theta_shape, np.double)
		
		# for x,y
		cdef object e_xy_dist = norm(0, e_xy)
		cdef double e_xy_max = e_xy_dist.pdf(0)
		for i in range(1, self.ground_map.shape[0]/2):
			e_i = e_xy_dist.pdf(self.dxyC2W(i))
			if e_i < self.max_prob_error * e_xy_max:
				break
		cdef int d_xy_range = i
		cdef int d_xy_shape = i*2 + 1 
		cdef np.ndarray[double, ndim=2] e_xy_p = np.empty([d_xy_shape, d_xy_shape], np.double)
		
		# pre-compute values for fast lookup in inner loop for theta
		d_theta_i = self.dthetaW2C(d_theta)
		d_theta_d = d_theta - self.dthetaC2W(d_theta_i)
		e_theta_dist = norm(d_theta_d, e_theta)
		for i in range(e_theta_p.shape[0]):
			e_theta_p[i] = e_theta_dist.pdf(self.dthetaC2W(i - e_theta_p.shape[0]/2))
		e_theta_p /= e_theta_p.sum()
		#print e_theta_p
		#print d_theta_i, d_theta_d
		
		# view to remove some safety checks in the inner-most loop
		cdef np.ndarray[double, ndim=3] PX_view = np.asarray(self.PX)
		# temporary storage for probability mass
		cdef np.ndarray[double, ndim=3] PX_new = np.zeros(np.asarray(self.PX).shape, np.double)
		
		# mass probability transfer loops, first iterate on theta on source cells
		for i in range(angle_N):
			# change in angle
			s_theta = self.thetaC2W(i)
			t_theta = s_theta + d_theta
			# rotation matrix for theta
			T = rot_mat2(t_theta)
			
			# compute displacement for this theta
			d_x_r, d_y_r = T.dot([d_x, d_y])
			d_x_r_i = self.dxyW2C(d_x_r)
			d_y_r_i = self.dxyW2C(d_y_r)
			d_x_r_d = d_x_r - self.dxyW2C(d_x_r_i)
			d_y_r_d = d_y_r - self.dxyW2C(d_y_r_i)
			
			# compute covariance
			sigma = T.dot(e_xy_mat).dot(T.transpose())
			
			# then pre-compute arrays for fast lookup in inner loop for x,y
			mu = np.array([d_x_r_d, d_y_r_d])
			#print 'mu', mu
			for j in range(e_xy_p.shape[0]):
				for k in range(e_xy_p.shape[1]):
					t_x = self.dxyC2W(j - e_xy_p.shape[0]/2)
					t_y = self.dxyC2W(k - e_xy_p.shape[1]/2)
					e_xy_p[j,k] = _norm_pdf_multivariate(np.array([t_x, t_y]), mu , sigma)
			e_xy_p /= e_xy_p.sum()
			#print e_xy_p
			#scipy.misc.imsave('/tmp/toto/e_xy_p-'+str(i)+'.png', e_xy_p)
			
			# outer loops for x,y iterating on source cells
			for j in range(w):
				for k in range(h):
					# inner loops iterating on target cells
					for d_i in range(d_theta_shape):
						u_theta_i = i + d_theta_i + d_i - d_theta_range
						for d_j in range(d_xy_shape):
							u_x_i = j + d_x_r_i + d_j - d_xy_range
							if u_x_i >= 0 and u_x_i < w:
								for d_k in range(d_xy_shape):
									u_y_i = k + d_y_r_i + d_k - d_xy_range
									if u_y_i >= 0 and u_y_i < h:
										u_theta_i = (u_theta_i + angle_N) % angle_N
										# copy probability mass
										PX_new[u_theta_i, u_x_i, u_y_i] += PX_view[i, j, k] * e_theta_p[d_i] * e_xy_p[d_j, d_k]
			
		# copy back probability mass
		self.PX = PX_new
	
	def estimate_state(self):
		""" return a (x,y,theta) numpy array representing the estimated state """
		
		theta_i, x_i, y_i = np.unravel_index(np.asarray(self.PX).argmax(), (<object>self.PX).shape)
		return np.array([self.xyC2W(x_i), self.xyC2W(y_i), self.thetaC2W(theta_i)])
	
	def estimate_logratio(self, double x, double y, double theta):
		""" return the log ratio between the probability at estimate and at given location (x,y,theta).
		No bound check is performed on input """
		log_estimate = log(np.asarray(self.PX).max())
		log_query = log(self.PX[self.thetaW2C(theta), self.xyW2C(x), self.xyW2C(y)])
		return log_estimate - log_query
		
	
	# debug methods
	
	def dump_obs_model(self, str base_filename):
		""" Write images of observation model """
		cdef int i
		for i in range(self.angle_N):
			scipy.misc.imsave(base_filename+'-'+str(i)+'-left_black.png', self.obs_left_black[i])
			scipy.misc.imsave(base_filename+'-'+str(i)+'-left_white.png', self.obs_left_white[i])
			scipy.misc.imsave(base_filename+'-'+str(i)+'-right_black.png', self.obs_right_black[i])
			scipy.misc.imsave(base_filename+'-'+str(i)+'-right_white.png', self.obs_right_white[i])
	
	def dump_PX(self, str base_filename, float x = -1, float y = -1, float theta = -1):
		""" Write images of latent space """
		
		# dump image in RGB
		def write_image(np.ndarray[double, ndim=2] array_2D, str filename):
			cdef np.ndarray[double, ndim=3] zeros = np.zeros([self.PX.shape[1], self.PX.shape[2], 1], np.double)
			array_rgb = np.concatenate((array_2D[:,:,np.newaxis], zeros, zeros), axis = 2)
			cdef int i_x = self.xyW2C(x)
			cdef int i_y = self.xyW2C(y)
			if self.is_in_bound_cell(i_x, i_y):
				#array_rgb[i_x,i_y,1] = array_rgb[i_x,i_y,0]
				#array_rgb[i_x,i_y,2] = array_rgb[i_x,i_y,0]
				max_value = array_2D.max()
				array_rgb[i_x,i_y,:] = [0,max_value,max_value]
			else:
				print 'WARNING: ground-truth position {},{} is outside map bounds'.format(x,y)
			scipy.misc.imsave(filename, array_rgb)
		
		# for every angle
		cdef int i
		for i in range(self.angle_N):
			write_image(np.asarray(self.PX[i]), base_filename+'-'+str(i)+'.png')
			
		# and the sum
		write_image(np.asarray(self.PX).sum(axis=0), base_filename+'-sum.png')
	
	# support methods
	
	cpdef double thetaC2W(self, int angle):
		""" Transform an angle in cell coordinates into an angle in radian """
		return ((angle+0.5) * 2. * _pi) / self.angle_N

	cpdef int thetaW2C(self, double angle):
		""" Transform an angle in radian into an angle in cell coordinates """
		return int(floor((angle * self.angle_N) / (2. * _pi))) % self.angle_N
	
	cpdef double dthetaC2W(self, int dangle):
		""" Transform an angle difference in cell coordinates into a difference in radian """
		return ((dangle) * 2. * _pi) / self.angle_N
	
	cpdef int dthetaW2C(self, double dangle):
		""" Transform an angle difference in radian into a difference in cell coordinates """
		return int(round((dangle * self.angle_N) / (2. * _pi)))

