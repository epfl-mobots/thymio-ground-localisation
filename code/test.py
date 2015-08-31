#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

# note: encode videos with
#       mencoder mf://PX-*-B_*-sum.png -ovc lavc -o sum.mp4

import pyximport; pyximport.install()
import numpy as np
import localize_with_cpt
import localize_with_montecarlo
from localize_with_cpt import rot_mat2
import math
from termcolor import colored
import argparse
import sys
import os.path
import scipy.misc

# support functions

def sensors_from_pos(x, y, theta):
	R = rot_mat2(theta)
	return R.dot([7.2, 1.1]) + [x,y], R.dot([7.2, -1.1]) + [x,y]

def normalize_angle(alpha):
	while alpha > math.pi:
		alpha -= 2 * math.pi
	while alpha < -math.pi:
		alpha += 2 * math.pi
	return alpha

def create_localizer(ground_map, args):
	if args.ml_angle_count:
		return localize_with_cpt.CPTLocalizer(ground_map, args.ml_angle_count, args.prob_correct, args.max_prob_error, args.prob_uniform, args.alpha_xy, args.alpha_theta)
	elif args.mcl_particles_count:
		return localize_with_montecarlo.MCLocalizer(ground_map, args.mcl_particles_count, args.prob_correct, args.prob_uniform, args.alpha_xy, args.alpha_theta)
	else:
		print 'You must give either one of --ml_angle_count or --mcl_particles_count argument to this program'
		sys.exit(1)

def dump_error(localizer, i, text, gt_x, gt_y, gt_theta, performance_log = None):
	estimated_state = localizer.estimate_state()
	dist_xy = np.linalg.norm(estimated_state[0:2]-(gt_x,gt_y))
	dist_theta = math.degrees(normalize_angle(estimated_state[2]-gt_theta))
	logratio_P = localizer.estimate_logratio(gt_x, gt_y, gt_theta)
	if abs(dist_xy) < math.sqrt(2)*2 and abs(dist_theta) < 15:
		color = 'green'
	elif abs(dist_xy) < math.sqrt(2)*4 and abs(dist_theta) < 30:
		color = 'yellow'
	else:
		color = 'red'
	print colored('{} {} - x,y dist: {}, theta dist: {}, log ratio P: {}'.format(i, text, dist_xy, dist_theta, logratio_P), color)
	if performance_log:
		performance_log.write('{} {} {} {} {} {} {} {} {}\n'.format(\
			gt_x, \
			gt_y, \
			gt_theta, \
			estimated_state[0], \
			estimated_state[1], \
			estimated_state[2], \
			dist_xy, \
			dist_theta, \
			logratio_P
		))


# main test function

def test_command_sequence(ground_map, localizer, sequence):
	""" Evaluate a sequence of positions (x,y,theta) """

	# initialise position
	x, y, theta = sequence.next()

	for i, (n_x, n_y, n_theta) in enumerate(sequence):
		# observation
		# get sensor values from gt
		lpos, rpos = sensors_from_pos(x, y, theta)
		lval = ground_map[localizer.xyW2C(lpos[0]), localizer.xyW2C(lpos[1])]
		rval = ground_map[localizer.xyW2C(rpos[0]), localizer.xyW2C(rpos[1])]
		# apply observation
		localizer.apply_obs(lval > 0.5, rval > 0.5)
		if args.debug_dump:
			localizer.dump_PX(os.path.join(args.debug_dump, 'PX-{:0>4d}-A_obs'.format(i)), x, y, theta)
		# compare observation with ground truth before movement
		dump_error(localizer, i, "after obs", x, y, theta)

		# compute movement
		d_theta = n_theta - theta
		d_x, d_y = rot_mat2(-theta).dot([n_x-x, n_y-y])
		# do movement
		x, y, theta =  n_x, n_y, n_theta
		localizer.apply_command(d_x, d_y, d_theta)
		if args.debug_dump:
			localizer.dump_PX(os.path.join(args.debug_dump, 'PX-{:0>4d}-B_mvt'.format(i)), x, y, theta)
		# compare observation with ground truth after movement
		dump_error(localizer, i, "after mvt", x, y, theta)


def self_test(args):
	""" Self tests """

	# trajectory generators

	def traj_linear_on_x(x_start, x_end, delta_x, y, d_t):
		""" generator for linear trajectory on x """
		for x in np.arange(x_start, x_end, delta_x):
			yield x, y, 0.

	def traj_linear_on_y(y_start, y_end, delta_y, x, d_t):
		""" generator for linear trajectory on x """
		for y in np.arange(y_start, y_end, delta_y):
			yield x, y, math.pi/2

	def traj_circle(x_center, y_center, radius, d_alpha, d_t):
		""" generator for circular trajectory """
		for alpha in np.arange(0, 2 * math.pi, d_alpha):
			x = x_center + math.cos(alpha) * radius
			y = y_center + math.sin(alpha) * radius
			yield x, y, alpha + math.pi/2

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

		# run test
		print title
		test_command_sequence(ground_map, create_localizer(ground_map, args), generator)
		print ''


def eval_data(args):
	# load image as ground map and convert to range 0.0:1.0
	ground_map = scipy.misc.imread(os.path.join(args.eval_data, 'map.png')).astype(float)
	ground_map /= ground_map.max()

	# build localizer
	localizer = create_localizer(ground_map, args)

	# if given, opens performance log
	if args.performance_log:
		performance_log = open(args.performance_log, 'w')
	else:
		performance_log = None

	frame_skip_counter = args.skip_start_frames
	o_odom_x, o_odom_y, o_odom_theta = None, None, None
	o_gt_x, o_gt_y, o_gt_theta = None, None, None
	for i, (gt_line, odom_pos_line, odom_quat_line, sensor_left_line, sensor_right_line) in enumerate(zip(\
		open(os.path.join(args.eval_data, 'gt.txt')), \
		open(os.path.join(args.eval_data, 'odom_pose.txt')), \
		open(os.path.join(args.eval_data, 'odom_quaternion.txt')), \
		open(os.path.join(args.eval_data, 'sensor_left.txt')), \
		open(os.path.join(args.eval_data, 'sensor_right.txt')) \
	)):
		#print gt_line, odom_pos_line, odom_quat_line, sensor_left_line, sensor_right_line
		gt_x, gt_y, gt_theta = map(float, gt_line.split())
		gt_x *= 100; gt_y *= 100
		#print 'gt', gt_x, gt_y, gt_theta
		odom_x, odom_y = map(float, odom_pos_line.split())
		odom_x *= 100; odom_y *= 100
		z, w = map(float, odom_quat_line.split())
		odom_theta = np.arcsin(z) * 2. * np.sign(w)
		#print odom_x, odom_y, odom_theta
		sensor_left = float(sensor_left_line.strip()) > 350
		sensor_right = float(sensor_right_line.strip()) > 350
		#print sensor_left, sensor_right

		# optionally skip frames
		if frame_skip_counter > 0:
			frame_skip_counter -= 1
			continue

		# if first line, just store first data for local frame computation
		if not o_odom_x:
			o_odom_x, o_odom_y, o_odom_theta = odom_x, odom_y, odom_theta
			o_gt_x, o_gt_y, o_gt_theta = gt_x, gt_y, gt_theta
			continue

		# TODO: add handling of skip_frames

		# else compute movement
		# odom
		odom_d_theta = odom_theta - o_odom_theta
		odom_d_x, odom_d_y = rot_mat2(-o_odom_theta).dot([odom_x-o_odom_x, odom_y-o_odom_y])
		o_odom_x, o_odom_y, o_odom_theta = odom_x, odom_y, odom_theta
		# ground truth
		gt_d_theta = gt_theta - o_gt_theta
		gt_d_x, gt_d_y = rot_mat2(-o_gt_theta).dot([gt_x-o_gt_x, gt_y-o_gt_y])
		o_gt_x, o_gt_y, o_gt_theta = gt_x, gt_y, gt_theta

		# do movement
		localizer.apply_command(odom_d_x, odom_d_y, odom_d_theta)
		dump_error(localizer, i, "after mvt", gt_x, gt_y, gt_theta, performance_log)
		if args.debug_dump:
			localizer.dump_PX(os.path.join(args.debug_dump, 'PX-{:0>4d}-A_mvt'.format(i)), gt_x, gt_y, gt_theta)
			print '  d_odom (local frame): ', odom_d_x, odom_d_y, odom_d_theta
			print '  d_gt (local frame):   ', gt_d_x, gt_d_y, gt_d_theta

		# apply observation
		localizer.apply_obs(sensor_left, sensor_right)
		if args.debug_dump:
			localizer.dump_PX(os.path.join(args.debug_dump, 'PX-{:0>4d}-B_obs'.format(i)), gt_x, gt_y, gt_theta)

	# close log file
	if performance_log:
		performance_log.close()


if __name__ == '__main__':

	# command line parsing
	parser = argparse.ArgumentParser(description='Test program for Thymio localization')
	parser.add_argument('--ml_angle_count', type=int, help='Use Markov localization with a discretized angle of angle_count')
	parser.add_argument('--mcl_particles_count', type=int, help='Use Monte Carlo localization with a particles_count particles')
	parser.add_argument('--self_test', help='run unit-testing style of tests on synthetic data', action='store_true')
	parser.add_argument('--eval_data', type=str, help='eval data from directory given as parameter')
	parser.add_argument('--prob_correct', type=float, default=0.95, help='probability when seeing a correct ground color (default: 0.95)')
	parser.add_argument('--max_prob_error', type=float, default=0.01, help='max. error ratio with mode value when spilling over probability in Markov localisation (default: 0.01)')
	parser.add_argument('--prob_uniform', type=float, default=0.05, help='uniform probability added to fight depletion (default: 0.05)')
	parser.add_argument('--alpha_xy', type=float, default=0.11, help='relative linear error in motion model (default: 0.11)')
	parser.add_argument('--alpha_theta', type=float, default=0.17, help='relative angular error in motion model (default: 0.17)')
	parser.add_argument('--debug_dump', type=str, help='directory where to dump debug information (default: do not dump)')
	parser.add_argument('--performance_log', type=str, help='filename in which to write performance log (default: do not write log)')
	parser.add_argument('--skip_start_frames', type=int, default=0, help='optionally, some frames to skip at the beginning of the data file (default: 0)')
	parser.add_argument('--skip_frames', type=int, default=0, help='optionally, some frames to skip when processing the data file (default: 0)')

	args = parser.parse_args()

	if args.self_test:
		self_test(args)
	elif args.eval_data:
		eval_data(args)
	else:
		print 'No task given, use either one of --self_test or --eval_data'
