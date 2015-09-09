#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

import os
import numpy as np
import matplotlib.pyplot as plt

result_base_dir = '../result'
data_base_dir = '../data'

def draw_plot(algo, runs, params, showDistNotAngle):

	# show dist or angle diff?
	if showDistNotAngle:
		dataCol = 8
		ylabel = 'distance to ground-truth position [cm]'
	else:
		dataCol = 9
		ylabel = 'distance to ground-truth orientation [degrees]'

	# load and average all given runs
	x_ticks = np.arange(0., 100.)
	for param in params:
		y_average_values = np.zeros(shape=x_ticks.shape, dtype=float)
		for run in runs:
			data = np.loadtxt(os.path.join(result_base_dir, '{}_{}_{}'.format(run, algo, param)))
			gt = np.loadtxt(os.path.join(data_base_dir, run, 'gt.txt'))
			indices = data[:,0].astype(int)
			xy = gt[indices,0:2]
			deltas_xy = np.linalg.norm(np.diff(xy, axis=0), axis=1)
			x_values = np.insert(np.cumsum(deltas_xy), 0, [0.]) * 100.
			y_values = np.abs(data[:,dataCol])
			y_average_values += np.interp(x_ticks, x_values, y_values)

		# plot
		y_average_values /= len(runs)
		plt.plot(x_ticks, y_average_values, label=str(param))
		plt.xlabel('path length')
		plt.ylabel(ylabel)

	# add legend and show plot
	plt.legend()
	plt.show()

if __name__ == '__main__':
	# draw plots
	draw_plot('mcl', ['random_1'], ['50k', '100k', '200k', '400k'], True)
	draw_plot('mcl', ['random_2'], ['50k', '100k', '200k', '400k'], True)
	draw_plot('mcl', ['random_1', 'random_2'], ['50k', '100k', '200k', '400k'], True)
	draw_plot('mcl', ['random_1', 'random_2'], ['50k', '100k', '200k', '400k'], False)
	draw_plot('ml', ['random_1', 'random_2'], [18, 36, 54, 72], True)
	draw_plot('ml', ['random_1', 'random_2'], [18, 36, 54, 72], False)

