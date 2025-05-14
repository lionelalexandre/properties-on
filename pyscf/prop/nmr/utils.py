#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:15:11 2025

@author: teodora
"""

import numpy

def diis(F_list, e_list, max_diis):
    """
    Perform DIIS to get a new 2D Fock matrix
    """
    
    if len(F_list) > max_diis:                                      
        F_list.pop(0) 
        e_list.pop(0) 
    
    B_dim = len(F_list) + 1
    B_matrix = numpy.empty((B_dim, B_dim))
    B_matrix[-1, :] = -1
    B_matrix[:, -1] = -1
    B_matrix[-1, -1] = 0

    for i in range(len(F_list)):
        for j in range(len(F_list)):
            B_matrix[i, j] = numpy.sum(e_list[i] * e_list[j])

    rhs = numpy.zeros((B_dim))
    rhs[-1] = -1

    try:
        coeff = numpy.linalg.solve(B_matrix, rhs)
    except numpy.linalg.LinAlgError:
        print("B_matrix is singular, using lstsq")
        coeff, *_ = numpy.linalg.lstsq(B_matrix, rhs, rcond=None)

    if numpy.any(numpy.isnan(coeff)) or numpy.allclose(coeff[:-1], 0):
        print("Bad diis coeff")
        return None

    F_new = numpy.zeros_like(F_list[0])
    for i in range(len(F_list)):
        F_new += coeff[i] * F_list[i]

    return F_new

def build_ab(S0, Sinv, F0, D0):
    
    I = numpy.eye(D0.shape[0])

    A = Sinv @ (I - 2 * S0 @ D0) @ F0
    B = F0 @ (I - 2 * D0 @ S0) @ Sinv
    
    return A, B
    
def build_q(Sinv, F0, D0, S1_n, F1_n):
        
    Q_n = Sinv @ S1_n @ D0 @ F0 @ Sinv \
            + Sinv @ F0 @ D0 @ S1_n @ Sinv \
            - D0 @ F1_n @ Sinv \
            - Sinv @ F1_n @ D0 \
            + 2 * D0 @ F1_n @ D0
            
    return Q_n

def gershgorin_max(W):
    n = W.shape[0]
    v = numpy.zeros(n) 
    for i in range(n):
        sum = 0
        for j in range(n):
            if (i != j):
                sum = sum + abs(W[i,j])
        v[i] = W[i,i] + sum   
    return v.max()

def gershgorin_min(W):
    n = W.shape[0]
    v = numpy.zeros(n)
    for i in range(n):
        sum = 0
        for j in range(n):
            if (i != j):
                sum = sum + abs(W[i,j])                
        v[i] = W[i,i] - sum   
    return v.min()

