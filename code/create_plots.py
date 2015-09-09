#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

import os
import numpy as np
import matplotlib
matplotlib.use("PDF") # do this before pylab so you don'tget the default back end.
import matplotlib.pyplot as plt
import prettyplotlib as ppl
#plt.style.use('ggplot')
import os.path


result_base_dir = '../result'
data_base_dir = '../data'
dest_base_dir = '../article/figures'

plot_params = {'font.size' : 8,
		  'legend.fontsize': 8,
		  'legend.frameon': True }

colors = ['#2c7bb6', '#abd9e9', '#fdae61', '#d7191c']

width_in = 3.6
aspect_ratio = 4./2.
height_in = width_in / aspect_ratio


def draw_plot(algo, runs, params, showDistNotAngle, name):

	plt.rcParams.update(plot_params)

	# load and average all given runs
	fig, ax = plt.subplots(figsize=(width_in, height_in))

	# show dist or angle diff?
	if showDistNotAngle:
		dataCol = 8
		ylabel = 'position error [cm]'
		ax.set_ylim(0, 30)
	else:
		dataCol = 9
		ylabel = 'angular error [degrees]'
		ax.set_ylim(0, 90)

	x_ticks = np.arange(0., 150.)
	for i, param in enumerate(params):
		y_average_values = np.zeros(shape=x_ticks.shape, dtype=float)
		for run in runs:

			data = np.loadtxt(os.path.join(result_base_dir, '{}_{}_{}'.format(run, algo, param)))
			gt = np.loadtxt(os.path.join(data_base_dir, run, 'gt.txt'))

			# get the indices in gt for every line in data
			indices = data[:,0].astype(int)

			# compute the local change of the sensors position between every line in data
			# we consider the center between the two centers, which is in (12,0) cm local frame
			thetas = gt[indices,2]
			sensor_local_poses = np.vstack((np.cos(thetas) * .12, np.sin(thetas) * .12))
			deltas_sensor_poses = np.diff(sensor_local_poses, axis=1).transpose()

			# compute the change of the position of the robot's center between every line in data
			xys = gt[indices,0:2]
			deltas_xy = np.diff(xys, axis=0)

			# compute the global distances traveled by the sensors between every line in data
			deltas_dists = np.linalg.norm(deltas_xy + deltas_sensor_poses, axis=1)

			# sum these distances to get the x axis
			x_values = np.insert(np.cumsum(deltas_dists), 0, [0.]) * 100.
			# get the y values directly from data
			y_values = np.abs(data[:,dataCol])
			# interpolate to put them in relation with other runs
			y_average_values += np.interp(x_ticks, x_values, y_values)

		# plot
		y_average_values /= len(runs)
		ppl.plot(ax, x_ticks, y_average_values, label=str(param), color=colors[i])

	# add label, legend and show plot
	plt.xlabel('path length')
	plt.ylabel(ylabel)
	ppl.legend(ax)
	fig.savefig(os.path.join(dest_base_dir, name), bbox_inches='tight', pad_inches=0.02)


if __name__ == '__main__':
	# draw plots

	# ML whole range
	draw_plot('ml', ['random_1'], [18, 36, 54, 72], True, 'ml-whole_random_1-xy.pdf')
	draw_plot('ml', ['random_1'], [18, 36, 54, 72], False, 'ml-whole_random_1-theta.pdf')
	draw_plot('ml', ['random_2'], [18, 36, 54, 72], True, 'ml-whole_random_2-xy.pdf')
	draw_plot('ml', ['random_2'], [18, 36, 54, 72], False, 'ml-whole_random_2-theta.pdf')

	# MCL whole range
	draw_plot('mcl', ['random_1'], ['50k', '100k', '200k', '400k'], True, 'mcl-whole_random_1-xy.pdf')
	draw_plot('mcl', ['random_1'], ['50k', '100k', '200k', '400k'], False, 'mcl-whole_random_1-theta.pdf')
	draw_plot('mcl', ['random_2'], ['50k', '100k', '200k', '400k'], True, 'mcl-whole_random_2-xy.pdf')
	draw_plot('mcl', ['random_2'], ['50k', '100k', '200k', '400k'], False, 'mcl-whole_random_2-theta.pdf')

	#draw_plot('mcl', ['random_1', 'random_2'], ['50k', '100k', '200k', '400k'], True)
	#draw_plot('mcl', ['random_1', 'random_2'], ['50k', '100k', '200k', '400k'], False)
	#draw_plot('ml', ['random_1', 'random_2'], [18, 36, 54, 72], True)
	#draw_plot('ml', ['random_1', 'random_2'], [18, 36, 54, 72], False)

