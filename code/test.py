#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal
# vim: ts=4:sw=4:noexpandtab

# note: encode videos with
#       mencoder mf://PX-*-B_mvt-sum.png -ovc lavc -o sum.mp4

import numpy as np
import pyximport; pyximport.install()
import localize_with_cpt
from localize_with_cpt import rot_mat2
import math

def sensors_from_pos(x, y, theta):
	R = rot_mat2(theta)
	return R.dot([7.2, 1.1]) + [x,y], R.dot([7.2, -1.1]) + [x,y]
	
def test_command_sequence(ground_map, localizer, sequence):
	""" Evaluate a sequence of positions (x,y,theta) """
	
	# initialise position
	t, x, y, theta = sequence.next()
	
	for i, (n_t, n_x, n_y, n_theta) in enumerate(sequence):
		# observation
		# get sensor values from gt
		lpos, rpos = sensors_from_pos(x, y, theta)
		lval = ground_map[localizer.xyW2C(lpos[0]), localizer.xyW2C(lpos[1])]
		rval = ground_map[localizer.xyW2C(rpos[0]), localizer.xyW2C(rpos[1])]
		# apply observation
		localizer.apply_obs(lval > 0.5, rval > 0.5)
		#localizer.dump_PX('/tmp/toto/PX-{:0>4d}-A_obs'.format(i), localizer.xyW2C(x), localizer.xyW2C(y))
		# compare observation with ground truth before movement
		print '{} after obs: {}'.format(i, np.linalg.norm(localizer.estimate_state()[0:2]-(x,y)))
		
		# compute movement
		d_t, d_theta = n_t - t, n_theta - theta
		d_x, d_y = rot_mat2(-theta).dot([n_x-x, n_y-y])
		# do movement
		t, x, y, theta =  n_t, n_x, n_y, n_theta
		localizer.apply_command(d_t, d_x, d_y, d_theta)
		#localizer.dump_PX('/tmp/toto/PX-{:0>4d}-B_mvt'.format(i), localizer.xyW2C(x), localizer.xyW2C(y))
		# compare observation with ground truth after movement
		print '{} after mvt: {}'.format(i, np.linalg.norm(localizer.estimate_state()[0:2]-(x,y)))


if __name__ == '__main__':

	# trajectory generators

	def traj_linear_on_x(x_start, x_end, delta_x, y, d_t):
		""" generator for linear trajectory on x """
		t = 0
		for x in np.arange(x_start, x_end, delta_x):
			yield t, x, y, 0.
			t += d_t
	
	def traj_linear_on_y(y_start, y_end, delta_y, x, d_t):
		""" generator for linear trajectory on x """
		t = 0
		for y in np.arange(y_start, y_end, delta_y):
			yield t, x, y, math.pi/2
			t += d_t
	
	def traj_circle(x_center, y_center, radius, d_alpha, d_t):
		""" generator for circular trajectory """
		t = 0
		for alpha in np.arange(0, 2 * math.pi, d_alpha):
			x = x_center + math.cos(alpha) * radius
			y = y_center + math.sin(alpha) * radius
			yield t, x, y, alpha + math.pi/2
			t += d_t
	
	# collection of generators
	generators = [
		("traj linear on x, y centered", traj_linear_on_x(5, 45, 1, 30, 1)),
		("traj linear on x, y low", traj_linear_on_x(5, 45, 1, 10, 1)),
		("traj linear on x, y high", traj_linear_on_x(5, 45, 1, 50, 1)),
		("traj linear on y, x centered", traj_linear_on_y(5, 45, 1, 30, 1)),
		("traj circle", traj_circle(30, 30, 20, math.radians(360/120), 1))
	]
	
	# ground map, constant
	ground_map = np.kron(np.random.choice([0.,1.], [20,20]), np.ones((3,3)))
	
	for title, generator in generators:
	
		# build localizer
		angle_N = 16
		prob_correct = 0.95
		localizer = localize_with_cpt.CPTLocalizer(ground_map, angle_N, prob_correct, 0.01)
		
		# run test
		print title
		test_command_sequence(ground_map, localizer, generator)
		print ''
		