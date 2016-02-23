#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal; remove-trailing-spaces all;
# vim: ts=4:sw=4:noexpandtab

import dbus
import time
import math

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
	if name == 'ground_values':
		dx_local = float(values[2]) * 0.01
		dy_local = float(values[3]) * 0.01
		dth_local = float(values[4]) * math.pi / 32767.
		x += dx_local * math.cos(th) - dy_local * math.sin(th)
		y += dx_local * math.sin(th) + dy_local * math.cos(th)
		th += dth_local
		if dx_local != 0.0 or dy_local != 0.0 or dth_local != 0.0:
			print x, y, th

if __name__ == '__main__':

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

	# run gobject loop
	loop = gobject.MainLoop()
	loop.run()

