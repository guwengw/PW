#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:58:27 2020

@author: lchapel
"""

import time
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist

import ot
import utils


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






def compute_perf_emd(dataset_p, dataset_u, n_pos, n_unl, prior, nb_reps,
                     nb_dummies=1):
    """Compute the performances of running the partial-W for a PU learning
    task on a given dataset several times

    Parameters
    ----------
    dataset_p: name of the dataset among which the positives are drawn

    dataset_u: name of the dataset among which the unlabeled are drawn

    n_pos: number of points in the positive dataset

    n_unl: number of points in the unlabeled dataset

    prior: percentage of positives on the dataset (s)

    nb_resp: number of runs

    nb_dummies: number of dummy points, default: no dummies
        (to avoid numerical instabilities of POT)

    Returns
    -------
    dict with:
        - the class prior
        - the performances of the p-w (avg among the repetitions)
        - the performances of the p-w with group constraints (avg)
        - the list of all the nb_reps performances of the p-w
        - the list of all the nb_reps performances of the p-w with groups
    """
    perfs = {}
    perfs['class_prior'] = prior
    perfs['emd'] = 0
    perfs['emd_groups'] = 0
    perfs_list = {}
    perfs_list['emd'] = []
    perfs_list['emd_groups'] = []
    start_time = time.time()
    for i in range(nb_reps):
        P, U, y_u = utils.draw_p_u_dataset_scar(dataset_p, dataset_u, n_pos,
                                                n_unl, prior, i)  # seed=i
        Ctot, _, _, mu, nu = compute_cost_matrices(P, U, prior, nb_dummies)
        nb_unl_pos = int(np.sum(y_u))

        transp_emd = ot.emd(mu, nu, Ctot)
        y_hat = np.ones(len(y_u))
        sum_dummies = np.sum(transp_emd[-nb_dummies:], axis=0)
        y_hat[np.argsort(sum_dummies)[nb_unl_pos:]] = 0
        perfs_list['emd'].append(np.mean(y_u == y_hat))
        perfs['emd'] += (np.mean(y_u == y_hat))

        transp_emd_group = pu_w_emd(mu, nu, Ctot, nb_dummies)
        y_hat = np.ones(len(y_u))
        sum_dummies = np.sum(transp_emd_group[-nb_dummies:], axis=0)
        y_hat[np.argsort(sum_dummies)[nb_unl_pos:]] = 0
        perfs_list['emd_groups'].append(np.mean(y_u == y_hat))
        perfs['emd_groups'] +=  (np.mean(y_u == y_hat))

    perfs['emd'] = perfs['emd'] / nb_reps
    perfs['emd_groups'] = perfs['emd_groups'] / nb_reps
    perfs['time'] = time.time() - start_time
    return perfs, perfs_list






