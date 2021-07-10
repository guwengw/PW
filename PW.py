# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 11:09:32 2021

@author: Administrator
"""

import utils
import numpy as np
import pandas as pd

import ot
import partial_gw as pgw
import matplotlib.pyplot as plt


n_unl = 800
n_pos = 400
nb_reps = 10
nb_dummies = 10

"""
prior = 0.518
perfs_mushrooms, perfs_list_mushrooms = pgw.compute_perf_emd('mushrooms', 'mushrooms', n_unl, n_pos, prior, nb_reps, nb_dummies)
avg_mush_emd_groups =  perfs_mushrooms['emd_groups']


prior = 0.786
perfs_shuttle, perfs_list_shuttle = pgw.compute_perf_emd('shuttle', 'shuttle', n_unl, n_pos, prior, nb_reps, nb_dummies)
avg_shut_emd_groups =  perfs_shuttle['emd_groups']



prior = 0.898
perfs_pageblocks, perfs_list_pageblocks = pgw.compute_perf_emd('pageblocks', 'pageblocks', n_unl, n_pos, prior, nb_reps, nb_dummies)
avg_page_emd_groups =  perfs_pageblocks['emd_groups']

prior = 0.167
perfs_usps, perfs_list_usps = pgw.compute_perf_emd('usps', 'usps', n_unl, n_pos, prior, nb_reps, nb_dummies)
avg_usps_emd_groups =  perfs_usps['emd_groups']

prior = 0.394
perfs_spambase, perfs_list_spambase = pgw.compute_perf_emd('spambase', 'spambase', n_unl, n_pos, prior, nb_reps, nb_dummies)
avg_spambase_emd_groups =  perfs_spambase['emd_groups']


prior = 0.5
perfs_house, perfs_list_house = pgw.compute_perf_emd('house', 'house', n_unl, n_pos, prior, nb_reps, nb_dummies)
avg_house_emd_groups =  perfs_house['emd_groups']
"""

prior = 0.5
perfs_mnist, perfs_list_mnist = pgw.compute_perf_emd('mnist', 'mnist', n_unl, n_pos, prior, nb_reps, nb_dummies)
avg_mnist_emd_groups =  perfs_mnist['emd_groups']