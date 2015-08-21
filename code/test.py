#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal
# vim: ts=4:sw=4:noexpandtab

import numpy as np
import pyximport; pyximport.install()
import localize_with_cpt
from localize_with_cpt import rot_mat2

def sensors_from_pos(x, y, theta):
	R = rot_mat2(theta)
	return R.dot([7.2, 1.1]) + [x,y], R.dot([7.2, -1.1]) + [x,y]

if __name__ == '__main__':
	ground_map = np.random.choice([0.,1.], [50,50])
	angle_N = 16
	prob_correct = 0.95
	localizer = localize_with_cpt.CPTLocalizer(ground_map, angle_N, prob_correct, 0.01)
	localizer.dump_obs_model('/tmp/toto/obs')
	#localizer.dump_PX('/tmp/toto/PX-0')
	#for i in range(100):
	#	print i
	#	localizer.apply_command(0.1, 0.0, 0.0, 0.1)
	#localizer.apply_command(1.0, 5.0, 0.0, 0.0)
	#localizer.dump_PX('/tmp/toto/PX-1')
	#localizer.apply_obs(True, True)
	#localizer.dump_PX('/tmp/toto/PX-2')
	
	theta = 0
	y = 25
	delta_x = 2
	for i, x in enumerate(np.arange(5, 35, delta_x)):
		# observation
		# get sensor values from gt
		lpos, rpos = sensors_from_pos(x, y, theta)
		#print x, y, theta
		#print lpos
		#print rpos
		lval = ground_map[localizer.xyW2C(lpos[0]), localizer.xyW2C(lpos[1])]
		rval = ground_map[localizer.xyW2C(rpos[0]), localizer.xyW2C(rpos[1])]
		#print lval, rval
		localizer.apply_obs(lval > 0.5, rval > 0.5)
		localizer.dump_PX('/tmp/toto/PX-'+str(i)+'-A_obs', localizer.xyW2C(x), localizer.xyW2C(y))
		# movement
		localizer.apply_command(1.0, delta_x, 0.0, 0.0)
		localizer.dump_PX('/tmp/toto/PX-'+str(i)+'-B_mvt', localizer.xyW2C(x), localizer.xyW2C(y))
		