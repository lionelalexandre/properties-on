#!/usr/bin/env python

# %%
'''

Computing NMR shielding constants
'''
import os

os.environ["PYSCF_EXT_PATH"] = "/Users/lioneltruflandier/pyscf-on/properties-on"

from pyscf import gto, dft, scf, lib
from pyscf.prop import nmr
from pyscf.data import nist
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

#%%
mol = gto.M(atom='''
            C 0 0 0
            O 0 0 1.1747            
            ''',
            basis='sto-3g', verbose=20)

M = mol.nao_nr()
Ne= mol.tot_electrons()
N = int(mol.tot_electrons()/2) # only for RHF

#%%
mf = scf.RHF(mol)
mf.conv_tol = 1e-14
mf.kernel()
#mf_nmr = nmr.RHF_DM(mf)
#mf_nmr.cphf = True
#mf_nmr.shielding()

#%%
# Get eigenstates from SCF
conv, e_tot, eigs, C, theta = scf.hf_dm.kernel(mf,conv_tol=1e-8)
C_occ = C[:,theta>0]

# Build Density matrix from C and theta
P     = mf.make_rdm1(mo_coeff=C, mo_occ=theta)

# Build H^{(10x)}
H_10 = nmr.rhf_dm.make_h10(mol,dm0=P)
H_10 = lib.einsum('xpq,pi,qj->xij', H_10, C.conj(), C_occ)

# Build S^{(10x)}
S_10 = nmr.rhf_dm.make_s10(mol)
S_10 = lib.einsum('xpq,pi,qj->xij', S_10, C.conj(), C_occ)

# Compute C^{(10x)}
mf_nmr = nmr.RHF_DM(mf)
mf_nmr.max_cycle_cphf = 5
mf_nmr.conv_tol       = 1e-3

# Gen vind function
fvind = nmr.rhf_dm.gen_vind(mf, mo_coeff=C, mo_occ=theta)

C_10, C_e10 = nmr.rhf_dm.solve_mo1(mf_nmr,
                                   mo_energy=eigs,
                                   mo_coeff=C, 
                                   mo_occ=theta,
                                   h1=H_10, 
                                   s1=S_10, 
                                   with_cphf=True)
print('separate call of fvind')
G_10 = fvind(mo1=C_10)
print('np.shape(G_10) =', np.shape(G_10), M)

#fvind_ao = nmr.rhf_dm.gen_vind_ao(mf, mo_coeff=C, mo_occ=theta)
#G_10_ao = fvind_ao(mo1=C_10)
#print('np.shape(G_10) =', np.shape(G_10_ao), M)



#mf = scf.RHF(mol)
#mf.conv_tol = 1e-14
#mf.kernel()
#mf_nmr = nmr.RHF_DM(mf)
#mf_nmr.cphf = True
#mf_nmr.shielding()



# solve_mo_1 > 
#    [call gen_vind]
#        call gen_response return (function)vresp (in _response_function)
#              call get_jk and return(function) vind 
#    [return (function)vind] 
"""
# Compute D^{(10x)}_{oo} and Compute D^{(10x)}_{vo}
occidx = theta > 0
viridx = theta == 0
C_occ  = C[:,occidx]
C_vir  = C[:,viridx]

D_10_oo = np.zeros((3,M,M))
D_10_vo = np.zeros((3,M,M))
for (X,i) in zip(C_10,range(3)):
    print(np.shape(X),np.shape(X[occidx]))
    D_10_oo[i] = np.matmul(C_occ,np.matmul(X[occidx],C_occ.T.conj()))    
    D_10_vo[i] = np.matmul(C_vir,np.matmul(X[viridx],C_occ.T.conj()))

# Duild D^{(10x)}_{ov} 
shielding_nuc = range(mol.natm)
para_vir = np.zeros((len(shielding_nuc),3,3))
para_occ = np.zeros((len(shielding_nuc),3,3))
D_10_ov = np.zeros((3,M,M))
for n, atm_id in enumerate(shielding_nuc):
    D_10_ov[n,:,:] = -D_10_vo[n,:,:].T.conj()


# Build the full perturbed density matrix
D_10 = 2*D_10_oo + D_10_ov + D_10_vo

"""
#mf_nmr = nmr.RHF(mf)
#mf_nmr.cphf = False
#mf_nmr.shielding()

#mf = scf.HF(mol)
#conv, e_tot, eigs, C, theta = scf.hf.kernel(mf,conv_tol=1e-8)

#C_10, C_e10 = nmr.rhf.solve_mo1(mf,
#                                   mo_energy=eigs,
#                                   mo_coeff=C, 
#                                   mo_occ=theta,
#                                   h1=H_10_, 
#                                   s1=S_10_, 
#                                   with_cphf=True)


"""
>>> from G09:
# HF/sto-3G NMR=GIAO MaxDisk=2GW

test

0 1
C 0. 0. 0.
O 0. 0. 1.1747
>>> get :
 SCF GIAO Magnetic shielding tensor (ppm):
  1  C    Isotropic =    -1.3194   Anisotropy =   405.6017            
   XX=  -136.5199   YX=     0.0000   ZX=     0.0000
   XY=     0.0000   YY=  -136.5199   ZY=     0.0000
   XZ=     0.0000   YZ=     0.0000   ZZ=   269.0818
   Eigenvalues:  -136.5199  -136.5199   269.0818
  2  O    Isotropic =   -20.2260   Anisotropy =   636.4890
   XX=  -232.3890   YX=     0.0000   ZX=     0.0000
   XY=     0.0000   YY=  -232.3890   ZY=     0.0000
   XZ=     0.0000   YZ=     0.0000   ZZ=   404.1000
   Eigenvalues:  -232.3890  -232.3890   404.1000
"""