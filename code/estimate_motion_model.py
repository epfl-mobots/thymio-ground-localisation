#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal
# vim: ts=4:sw=4:noexpandtab

import numpy as np
import matplotlib.pyplot as plt
import math
import os.path
import sys
import scipy.stats
import nlopt

# support functions

def rot_mat2(angle):
	""" Create a 2D rotation matrix for angle """
	return np.array([[np.cos(angle), -np.sin(angle)],
	                 [np.sin(angle),  np.cos(angle)]])

def normalize_angle(alpha):
	while alpha > math.pi:
		alpha -= 2. * math.pi
	while alpha < -math.pi:
		alpha += 2. * math.pi
	return alpha

# taken from http://stackoverflow.com/questions/11615664/multivariate-normal-density-in-python
def _norm_pdf_multivariate(x, sigma):
	""" multivariate PDF for Gaussian """
	size = len(x)
	if (size, size) == sigma.shape:
		det = np.linalg.det(sigma)
		if det == 0:
			raise NameError("The covariance matrix can't be singular")

		norm_const = 1.0/ ( math.pow((2*math.pi),float(size)/2) * math.pow(det,1.0/2) )
		x_mu = np.matrix(x) # mu is zero
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


# main functions

# lists of deltas, to be used within the optimisation function
odom_d_xs, odom_d_ys, odom_d_thetas = [], [], []
gt_d_xs, gt_d_ys, gt_d_thetas = [], [], []

def load_data(data_dirs):
	""" Load deltas x/y/theta for odom/ground-truth from all given data directories """
	
	for data_dir in data_dirs:
	
		# temporary variables
		o_odom_x, o_odom_y, o_odom_theta = None, None, None
		o_gt_x, o_gt_y, o_gt_theta = None, None, None
		
		# load all lines
		for i, (gt_line, odom_pos_line, odom_quat_line) in enumerate(zip(\
			open(os.path.join(data_dir, 'gt.txt')), \
			open(os.path.join(data_dir, 'odom_pose.txt')), \
			open(os.path.join(data_dir, 'odom_quaternion.txt')) \
		)):
			gt_x, gt_y, gt_theta = map(float, gt_line.split())
			gt_x *= 100; gt_y *= 100
			odom_x, odom_y = map(float, odom_pos_line.split())
			odom_x *= 100; odom_y *= 100
			z, w = map(float, odom_quat_line.split())
			odom_theta = np.arcsin(z) * 2. * np.sign(w)
			
			# if first line, just store first data for local frame computation
			if not o_odom_x:
				o_odom_x, o_odom_y, o_odom_theta = odom_x, odom_y, odom_theta
				o_gt_x, o_gt_y, o_gt_theta = gt_x, gt_y, gt_theta
				continue
			
			# else compute movement
			# odom
			odom_d_theta = odom_theta - o_odom_theta
			odom_d_x, odom_d_y = rot_mat2(-o_odom_theta).dot([odom_x-o_odom_x, odom_y-o_odom_y])
			odom_d_xs.append(odom_d_x)
			odom_d_ys.append(odom_d_y)
			odom_d_thetas.append(odom_d_theta)
			o_odom_x, o_odom_y, o_odom_theta = odom_x, odom_y, odom_theta
			# ground truth
			gt_d_theta = gt_theta - o_gt_theta
			gt_d_x, gt_d_y = rot_mat2(-o_gt_theta).dot([gt_x-o_gt_x, gt_y-o_gt_y])
			gt_d_xs.append(gt_d_x)
			gt_d_ys.append(gt_d_y)
			gt_d_thetas.append(gt_d_theta)
			o_gt_x, o_gt_y, o_gt_theta = gt_x, gt_y, gt_theta 


def compute_log_likelihood(params, grad):
	""" Sums the log-likelihood of the Gaussian error for all deltas """
	
	# parameters to test
	alpha_theta_to_xy = params[0]
	alpha_xy_to_xy = params[1]
	alpha_theta_to_theta = params[2]
	alpha_xy_to_theta = params[3]
	
	print 'alpha_theta_to_xy:', alpha_theta_to_xy
	print 'alpha_xy_to_xy:', alpha_xy_to_xy
	print 'alpha_theta_to_theta:', alpha_theta_to_theta
	print 'alpha_xy_to_theta:', alpha_xy_to_theta

	# temporary variables
	X = np.empty([2])
	sigma = np.zeros([2,2])
	
	log_likelihood = 0.
	
	# for every time steps
	for odom_d_x, odom_d_y, odom_d_theta, gt_d_x, gt_d_y, gt_d_theta in zip(\
		odom_d_xs, \
		odom_d_ys, \
		odom_d_thetas, \
		gt_d_xs, \
		gt_d_ys, \
		gt_d_thetas \
	):
		# compute SD for Gaussian error model
		norm_xy = math.sqrt(odom_d_x*odom_d_x + odom_d_y*odom_d_y)
		e_theta = alpha_xy_to_theta * norm_xy + alpha_theta_to_theta * math.fabs(odom_d_theta) + math.radians(0.5)
		e_xy = alpha_xy_to_xy * norm_xy + alpha_theta_to_xy * math.fabs(odom_d_theta) + 0.01
		
		# evaluate likelihood for this observation
		
		# theta
		dd_theta = normalize_angle(gt_d_theta - odom_d_theta)
		lh_theta = scipy.stats.norm.pdf(dd_theta, scale=e_theta)
		
		# x,y
		X[0] = gt_d_x - odom_d_x
		X[1] = gt_d_y - odom_d_y
		sigma[0,0] = e_xy
		sigma[1,1] = e_xy
		lh_xy = _norm_pdf_multivariate(X, sigma)
		
		log_likelihood += math.log(lh_theta) + math.log(lh_xy)
	
	print 'log likelihood:', log_likelihood, '\n'
	return log_likelihood


if __name__ == '__main__':

	# parse command line
	if len(sys.argv) == 1:
		print 'Usage: ' + sys.argv[0] + ' DATA_DIR [DAT_DIR]+'
		exit(1)
	else:
		data_dirs = sys.argv[1:]
	
	# load data from a list of directories
	print 'Evaluating likelihood on ' + str(data_dirs)
	load_data(data_dirs)
	
	# setup and run global optimisation
	# Main variants of possible global optimisation algorithms:
	# GN_DIRECT_L, GN_CRS2_LM, G_MLSL_LDS, GD_STOGO, GN_ISRES, GN_ESCH
	opt = nlopt.opt(nlopt.GN_DIRECT_L, 4)
	opt.set_lower_bounds([0.,0.,0.,0.])
	opt.set_upper_bounds([1.,1.,1.,1.])
	opt.set_max_objective(compute_log_likelihood)
	opt.set_xtol_abs(1e-3) # parameters change less than 0.1 %
	opt.set_ftol_rel(1e-5) # LL changes less than 0.01 %
	xopt = opt.optimize([0.1, 0.1, 0.1, 0.1])
	print "optimum at ", xopt
	print "minimum value = ", opt.last_optimum_value()
	print "result code = ", opt.last_optimize_result()
	