#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:29:47 2025

@author: teodora
"""

import numpy

def d1(mo_energy, mo_occ, mo_coeff, h1, s1):
    """
    Dodds J. L., McWeeny R., Sadlej A. J. Mol. Phys., Vol. 34, No. 6, 1977, 1779–1791.
    
    Equations (28) and (29);  R->D;  K->i;  L->a;
    
    D1 = -D(0)S(1)D(0) + x       
    D1 = -D(0)S(1)D(0) + sum( frac{(C_i)^†(h(1))-e_iS(1))C_a}{e_1 - e_a} (C_i(C_a)^† + C_a(C_i)^†) ) 
    
    """
    
    e_a = mo_energy[mo_occ==0]                                      #virtual orbital mo_occ==0 energies 
    e_i = mo_energy[mo_occ>0]                                       #occupied orbital energies 
    e_ai = 1 / (e_a.reshape(-1,1) - e_i)                            #inverse of the energy gap between unoccupied and occupied
    

                                                                    #building C_i for occupied and C_a for virtual coefficients
    C_i = mo_coeff[:, mo_occ > 0]                                   #shape: (ao, occ)
    C_a = mo_coeff[:, mo_occ == 0]                                  #shape: (ao, virt)
    
    print("C_i", C_i)
    print("C_a", C_a)
                                                                    #H_S = (h(1) - e_iS(1)) for each occupied i
    hs = h1 - s1 * e_i
    
    print("Shape of C_i.conj().T:", C_i.conj().T.shape)
    print("Shape of hs:", hs.shape)
    print("Shape of C_a:", C_a.shape)

                                                                    #computing nominator C_i^† (H(1)-e_iS(1)) C_a
    
    N_ia = numpy.einsum('ni,mnk,na->ia', C_i.conj(), hs, C_a)       #shape: (occ, virt)
       
                                                                    #scaling by the inverse of the energy gap 
    weighted_N_ia = N_ia * e_ai                                     #shape: (occ, virt)
    
    print("Shape of weighted_N_ia:", weighted_N_ia.shape)
    print("Shape of C_i:", C_i.shape)
    print("Shape of C_a.conj():", C_a.conj().shape)
    
    x = (   numpy.einsum('ij,mi,nj->mn', weighted_N_ia, C_i, C_a.conj()) +   
            numpy.einsum('ij,mi,nj->mn', weighted_N_ia, C_a, C_i.conj())    )  
            
                                                                    #shape: (ao,ao)
    
    d0 = numpy.einsum('ik,jk->ij', C_i, C_i.conj()) *2              #or alternatively d01 = numpy.dot(C_i, C_i.T) *2
    dsd = numpy.einsum('im,pmq,ma->ia', d0, s1, d0)                 

    D1 = -dsd - x                                                   #first order perturbed density matrix in ao basis (should be D_ov?)
    
    return dsd, x, D1    