#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 16:34:21 2025

@author: teodora
"""

import numpy as np

def build_huckel(n, alpha=5.0, beta=-1.0):
    H = np.zeros((n,n))
    np.fill_diagonal(H, alpha)
    for i in range (n):
        H[i, (i+1) % n ] = beta
        H[(i+1) % n, i] = beta
    return H

H = build_huckel(5)             # matrix H = [alpha beta   0    0    0   beta
                                          #   beta alpha  beta  0    0    0
                                          #     0   beta alpha beta  0    0
                                          #     0    0   beta alpha beta  0
                                          #     0    0     0 beta alpha beta 
                                          #   beta   0     0   0   beta alpha]
                
                
def get_system(n_orb, n_elec, alpha=5.0, beta=-1.0):
    F0 = build_huckel(n_orb, alpha=alpha, beta=beta)
    S0 = np.eye(n_orb)              

    eps, C = np.linalg.eigh(F0)
    n_occ = n_elec // 2
    C_occ = C[:, :n_occ]
    
    D0 = C_occ @ C_occ.T
    N = n_occ
    return F0, S0, D0, N


def dummy_F1_S1(F0, scale=1e-3, seed=1):
    rng = np.random.default_rng(seed)
    M = F0.shape[0]
    s = scale * np.linalg.norm(F0, ord='fro') / np.sqrt(M) # to keep perturbation small 

    F1 = np.empty((3, M, M))
    for k in range(3):
        A = rng.standard_normal((M, M))
        F1[k] = 0.5 * (A + A.T)
        F1[k] *= s / np.linalg.norm(F1[k], ord='fro')
        
    S1 = np.zeros_like(F1)
    return F1, S1


def overlap(M, kappa=1e8, seed=1):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((M, M)))   
    w = np.logspace(0, np.log10(kappa), M)  # eigenvalues: 1 ... kappa
    S = Q @ np.diag(w) @ Q.T
    S = 0.5*(S + S.T)
    return S


F0, S0, D0, N = get_system(10, 10)
F1, S1 = dummy_F1_S1(F0)
S0 = overlap(10)