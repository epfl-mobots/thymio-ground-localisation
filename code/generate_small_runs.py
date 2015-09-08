#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

import os

if __name__ == '__main__':
	try:
		os.mkdir('../result/small_runs')
	except OSError as e:
		pass
	runs = ['random_1', 'random_2']
	ml_params = [18, 36, 54, 72]
	mcl_params = [5, 10, 20, 50, 100, 150, 200, 400]
	for i in range(0,100,20):
		for run in runs:
#			for ml_param in ml_params:
#				command = './test.py --eval_data ../data/{} --ml_angle_count {} --performance_log ../result/small_runs/{}_ml_{}_{} --skip_at_start {} --duration 30'.format(run, ml_param, run, ml_param, i, i)
#				print 'Executing:', command
#				os.system(command)
			for mcl_param in mcl_params:
				command = './test.py --eval_data ../data/{} --mcl_particles_count {} --performance_log ../result/small_runs/{}_mcl_{}k_{} --skip_at_start {} --duration 30'.format(run, mcl_param*1000, run, mcl_param, i, i)
				print 'Executing:', command
				os.system(command)
