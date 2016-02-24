#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

import dbus
import time
import math

import matplotlib
#matplotlib.use('GTK')
import matplotlib.pyplot as plt
import numpy
import scipy
import scipy.misc

import sys

import pyximport; pyximport.install()
import localize_with_cpt

from dbus.mainloop.glib import DBusGMainLoop
import gobject
# required to prevent the glib main loop to interfere with Python threads
gobject.threads_init()
dbus.mainloop.glib.threads_init()

x = 0.
y = 0.
th = 0.


def ground_values_received(id, name, values):
	global x, y, th
	global odom_plot
	global localizer
	global prob_img
	global orient_plot
	if name == 'ground_values':
		# sensors
		sensor_left = float(values[0]) * 0.001 # input to the localizer is within 0 to 1
		sensor_right = float(values[1]) * 0.001 # input to the localizer is within 0 to 1
		# odometry
		dx_local = float(values[2]) * 0.01 * 0.1 # input to the localizer is in cm
		dy_local = float(values[3]) * 0.01 * 0.1 # input to the localizer is in cm
		dth_local = float(values[4]) * math.pi / 32767. # input to the localizer is in radian
		x += dx_local * math.cos(th) - dy_local * math.sin(th)
		y += dx_local * math.sin(th) + dy_local * math.cos(th)
		th += dth_local
		#if dx_local != 0.0 or dy_local != 0.0 or dth_local != 0.0:
		print sensor_left, sensor_right, x, y, th

		#odom_plot.set_xdata(numpy.append(odom_plot.get_xdata(), x))
		#odom_plot.set_ydata(numpy.append(odom_plot.get_ydata(), y))

		# localization
		print 'a'
		localizer.apply_command(dx_local, dy_local, dth_local)
		print 'b'
		localizer.apply_obs(sensor_left, sensor_right)
		print 'c'
		PX = localizer.get_PX().sum(axis=0)
		PX = PX * 255. / PX.max()
		print PX.shape
		prob_img.set_data(numpy.transpose(PX))
		print 'd'
		est_x, est_y, est_theta, conf = localizer.estimate_state()
		#print PX
		orient_plot.set_offsets(numpy.column_stack((
			[est_x, est_x + 3 * math.cos(est_theta)],
			[est_y, est_y + 3 * math.sin(est_theta)]
		)))
		plt.draw()


if __name__ == '__main__':

	# load stuff
	ground_map = numpy.flipud(scipy.misc.imread(sys.argv[1]).astype(float))
	#scipy.misc.imsave('/tmp/dump.png', numpy.transpose(ground_map))
	localizer = localize_with_cpt.CPTLocalizer(numpy.transpose(ground_map) / 255., 24, 0.15, 0.01, 0, 0.1, 0.1)

	# Glib main loop
	DBusGMainLoop(set_as_default=True)

	# open DBus
	bus = dbus.SessionBus()

	# get Aseba network
	try:
		network = dbus.Interface(
		bus.get_object('ch.epfl.mobots.Aseba', '/'),  dbus_interface='ch.epfl.mobots.AsebaNetwork')
	except dbus.exceptions.DBusException:
		raise AsebaException('Can not connect to Aseba DBus services! Is asebamedulla running?')

	# create filter for our event
	eventfilter = network.CreateEventFilter()
	events = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', eventfilter), dbus_interface='ch.epfl.mobots.EventFilter')
	events.ListenEventName('ground_values')
	dispatch_handler = events.connect_to_signal('Event', ground_values_received)

	# matplotlib init
	plt.ion()
	plt.figure()
	# maps
	plt.subplot(1, 2, 1)
	plt.imshow(ground_map, origin='lower', interpolation="nearest", cmap='gray')
	plt.subplot(1, 2, 2)
	prob_img = plt.imshow(ground_map, origin='lower', interpolation="nearest", cmap='gray')
	orient_plot = plt.scatter([0,3],[0,0], c=['#66ff66', '#ff4040'])
	#plt.subplot(1, 3, 3)
	plt.draw()

	# run gobject loop
	loop = gobject.MainLoop()
	loop.run()

