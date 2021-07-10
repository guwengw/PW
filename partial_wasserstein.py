# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:31:13 2021

@author: Administrator
"""



import numpy as np
import scipy as sp


import ot
import utils

import KM



from sklearn.metrics import f1_score


def compute_cost_matrices(P, U, prior, nb_dummies=0):

    # Positive dataset with dummy points
    n_unl_pos = int(U.shape[0]*prior)
    P_ = P.copy()
    P_ = np.vstack([P_, np.zeros((nb_dummies, P.shape[1]))])

    # weigths
    mu = (np.ones(len(P_))/(len(P_)-nb_dummies))*(n_unl_pos/len(U))
    if nb_dummies > 0:
        mu[-nb_dummies:] = (1 - np.sum(mu[:-nb_dummies]))/nb_dummies
    else:
        mu = mu / np.sum(mu)
    nu = np.ones(len(U))/len(U)

    # intra-domain
    C1 = sp.spatial.distance.cdist(P_, P_)
    C2 = sp.spatial.distance.cdist(U, U)
    if nb_dummies > 0:
        C1[:, -nb_dummies:] = C1[-nb_dummies:, :] = C2.max()*1e2
        C1[-nb_dummies:, -nb_dummies:] = 0

    # inter-domain
    if P_.shape[1] == U.shape[1]:
        C = sp.spatial.distance.cdist(P_, U)
        if nb_dummies > 0:
            C[-nb_dummies:, :] = 1e2 * C[:-nb_dummies, :].max()
    else:
        C = None
    return C, C1, C2, mu, nu


def pu_w_emd(p, q, C, nb_dummies=1):

    lstlab = np.array([0, 1])
    labels_a = np.append(np.array([0]*(len(p)-nb_dummies)),
                         np.array([1]*(nb_dummies)))

    def f(G):
        res = 0
        for i in range(G.shape[1]):
            for lab in lstlab:
                temp = G[labels_a == lab, i]
                res += (np.linalg.norm(temp, 1))**0.5
        return res

    def df(G):
        W = np.zeros(G.shape)
        for i in range(G.shape[1]):
            for lab in lstlab:
                temp = G[labels_a == lab, i]
                W[labels_a == lab, i] = 0.5*(np.linalg.norm(temp, 1))**(-0.5)
        return W

    Gc = ot.optim.cg(p, q, C, 1e6, f, df, numItermax=20)
    return Gc


n_unl = 800
n_pos = 400
nb_dummies = 10

prior = 0.5

dataset = "house"

dataset_p = dataset
dataset_u = dataset

P, U, y_u = utils.draw_p_u_dataset_scar(dataset_p, dataset_u, n_pos,
                                                n_unl, prior, 1)  

estimation_prior = KM.estimate_class_prior(P,U)

Ctot, _, _, mu, nu = compute_cost_matrices(P, U, estimation_prior, nb_dummies)
#nb_unl_pos = int(np.sum(y_u))
nb_unl_pos = int(estimation_prior*n_unl)

transp_emd = ot.emd(mu, nu, Ctot)
y_hat = np.ones(len(U))
sum_dummies = np.sum(transp_emd[-nb_dummies:], axis=0)
y_hat[np.argsort(sum_dummies)[nb_unl_pos:]] = 0

f1emd = f1_score(y_u,y_hat,average = 'micro')


transp_emd_group = pu_w_emd(mu, nu, Ctot, nb_dummies)
y_hat = np.ones(len(U))
sum_dummies = np.sum(transp_emd_group[-nb_dummies:], axis=0)
y_hat[np.argsort(sum_dummies)[nb_unl_pos:]] = 0
        
f1groups = f1_score(y_u,y_hat,average = 'micro')