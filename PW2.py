# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 17:23:53 2021

@author: Administrator
"""


import utils
import KM
import numpy as np
import pandas as pd
import codecs

import ot
import partial_gw as pgw
import matplotlib.pyplot as plt


from sklearn.metrics import f1_score

n_unl = 800
n_pos = 400
nb_dummies = 10

prior = 0.7

algorithm = "Partial Wasserestein"


dataset = "fashion"

dataset_p = dataset
dataset_u = dataset

P, U, y_u = utils.draw_p_u_dataset_scar(dataset_p, dataset_u, n_pos,
                                                n_unl, prior, 1)  

estimation_prior = KM.estimate_class_prior(P,U)

Ctot, _, _, mu, nu = pgw.compute_cost_matrices(P, U, estimation_prior, nb_dummies)
nb_unl_pos = int(np.sum(y_u))

transp_emd = ot.emd(mu, nu, Ctot)
y_hat = np.ones(len(y_u))
sum_dummies = np.sum(transp_emd[-nb_dummies:], axis=0)
y_hat[np.argsort(sum_dummies)[nb_unl_pos:]] = 0

f1emd = f1_score(y_u,y_hat,average = 'micro')


transp_emd_group = pgw.pu_w_emd(mu, nu, Ctot, nb_dummies)
y_hat = np.ones(len(y_u))
sum_dummies = np.sum(transp_emd_group[-nb_dummies:], axis=0)
y_hat[np.argsort(sum_dummies)[nb_unl_pos:]] = 0

errorPW = len(np.where((y_hat == y_u ) == False)[0]) 
errorRatePW = errorPW / y_u.shape[0]
        
f1groups = f1_score(y_u,y_hat,average = 'micro')

with codecs.open('F:/PU/experiment/result.txt',mode='a') as file_txt:
    file_txt.write(algorithm+'\t'+dataset+'\t'+"prior = "+str(prior)+'\t'+"estimation_prior = "+str(estimation_prior)+'\t'
                   +"error = "+str(errorRatePW)+'\t'+"F_score is "+str(f1groups)+'\n')
