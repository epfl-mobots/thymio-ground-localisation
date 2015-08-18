#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal
# vim: ts=4:sw=4:noexpandtab

import numpy as np
import pyximport; pyximport.install()
import localize_with_cpt

if __name__ == '__main__':
	ground_map = np.random.choice([0.,1.], [50,50])
	angle_N = 12
	prob_correct = 0.95
	localizer = localize_with_cpt.CPTLocalizer(ground_map, angle_N, prob_correct)
	localizer.dump_obs_model('/tmp/toto/obs')