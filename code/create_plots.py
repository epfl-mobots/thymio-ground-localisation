#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	result_base_dir = '../result'
	data_base_dir = '../data'

	#mcl_params = [5, 10, 20, 50, 100, 150, 200, 400]
	mcl_params = [10, 20, 50, 100, 150, 200, 400]
	runs = ['random_1', 'random_2']
	x_ticks = np.arange(0., 140.)

	for mcl_param in mcl_params:
		y_average_values = np.zeros(shape=x_ticks.shape, dtype=float)
		for run in runs:
			data = np.loadtxt(os.path.join(result_base_dir, '{}_mcl_{}k'.format(run, mcl_param)))
			gt = np.loadtxt(os.path.join(data_base_dir, run, 'gt.txt'))
			indices = data[:,0].astype(int)
			xy = gt[indices,0:2]
			deltas_xy = np.linalg.norm(np.diff(xy, axis=0), axis=1)
			x_values = np.insert(np.cumsum(deltas_xy), 0, [0.]) * 100.
			y_values = data[:,8]
			y_average_values += np.interp(x_ticks, x_values, y_values)

		# plot
		y_average_values /= len(runs)
		plt.plot(x_ticks, y_average_values, label=str(mcl_param))
		plt.xlabel('path length')
		plt.ylabel('distance to ground-truth position')

	# add legend and show plot
	plt.legend()
	plt.show()
