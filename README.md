Bayesian global localisation for Thymio using ground sensors
============================================================

This repository holds the source code, recorded VICON data, and submitted ICRA2016 paper.

Shiling Wang, Francis Colas, Ming Liu, Francesco Mondada, Stéphane Magnenat

How to use
----------

This code needs Python 2.x, Cython, numpy, scipy, and termcolor.

Once these are installed, you can play with existing datasets (in `data/`) using the `test.py` tool. If you launch it with the `-h` argument from the `data/` directory, it will list its available options.

For example:

    ./test.py --self_test --ml_angle_count 32

runs the synthetic data self tests using Markov localization and 32 discretization for angle.

The command used to generate the video for MCL being:

    ./test.py --eval_data ../data/random_long/ --skip_at_start 180 --duration 200 --prob_uniform 0.1 --custom_map ../data/kandinsky_comp-8.png --fake_observations --sigma_obs 0.1 --mcl_particles_count 200000 --debug_dump ../result/video/mcl --performance_log ../result/video/mcl/log

The rest of this section lists the different options.

Algorithm
---------

There are two algorithms, Markov localization (ML) and Monte Carlo localization (MCL).

ML can be selected by this switch:

    --ml_angle_count
  
A discretization for angles must be given as parameter.

MCL can be selected by this switch:

    --mcl_particles_count

A number of particles must be given as parameter.

Evaluations
-----------

Two types of evaluations can be done with this tool, self test and data evaluation.

Self test can be selected by this switch:

    --self_test

This will test the algorithm on noiseless synthetic data. If a set of parameters do not work with this mode, it is useless to try with real data.

Data evaluation can be selected by this switch:

    --eval_data

A directory from which to evaluate data must be given.

Getting information out of the tests
------------------------------------

In addition to observing the colors in the console, there are two ways to observe the performance of the test.

A log of the test can be requested with `--performance_log`, the argument being the file for logging. In that file, each line will have similar informations for every step.

In addition, images can be dumped with the `--debug_dump option`, taking a directory (hopefully empty) as parameter. The exact information depend on the algorithm, bitmaps are dumped for ML and maps of arrows for MCL.

Parameters
----------

There are different parameters. The main being the arguments to `--ml_angle_count` and `--mcl_particles_count.` In addition, there are these categories:

1. motion model and uniform probabilities: `--alpha_xy`, `--alpha_theta` and `--prob_uniform`. The current defaults have been found by maximum likelihood estimation on data sets "random_1", "random_2" and "random_long".

2. observation model: `--sigma_obs` the standard deviation of the observation model for ground color.

3. implementation cut-off for ML: `--max_prob_error`, can be ignored, but it might be set higher to improve speed at the expense of the accuracy of the probability tables computation.

Controlling which part of the dataset we use
--------------------------------------------

Two additional options allows to control which part of the dataset on want to use.

With `--skip_steps`, only one line every N steps will be used from the dataset. Therefore, we can simulate longer time steps. For instance `--skip_steps 10` will simulate observations only every 1 second.

With `--skip_at_start`, a certain number of steps can be skipped in the beginning of the file, because in some the robot is initially idle. If combined with the previous parameter, its effect will be multiplicative.

With `--duration`, only a certain number of steps will be used. If combined with the previous parameter, its effect will be multiplicative.

Custom map
----------

With `--custom_map`, a custom map can be used. By default `map.png` in the data directory will be used.

Faking observations
-------------------

To use grayscale maps that were not available at record time, the switch `--fake_observations` allows to regenerate observations from map and ground truth.
