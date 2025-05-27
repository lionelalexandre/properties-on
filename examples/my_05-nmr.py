#!/usr/bin/env python

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

mol = gto.M(atom='''
            C 0 0 0
            O 0 0 1.1747
            ''',
            basis='sto-3g', verbose=20)

M = mol.nao_nr()
Ne= mol.tot_electrons()
N = int(mol.tot_electrons()/2) # only for RHF
print('### M (basis set size), Ne (nb. of electrons), N (nb. of occupied states)')
print(M,Ne,N)

##########################################################
# C \in R^{(M\times M)} = unperturbed matrix of MO coeff.
##########################################################
mf = scf.hf_dm.SCF(mol)
conv, e_tot, eigs, C, theta = scf.hf_dm.kernel(mf,conv_tol=1e-14)
print(e_tot)

##########################################################
# C \in R^{(M\times M)} = unperturbed matrix of MO coeff.
##########################################################
print('np.shape(C) =', np.shape(C), M)

##########################################################
# Build P = 2\sum_{i\in occ} C_iC_i^{\dagger} with C_i column matrix (AO basis)
# Note: P = 2*D
##########################################################
# just call make_rdm1 from SCF
P_pyscf = mf.make_rdm1(mo_coeff=C, mo_occ=theta)

# hard coding from pyscf
C_occ = C[:,theta>0]
P_pyscf_ = (C_occ*theta[theta>0]).dot(C_occ.conj().T)

# soft coding
P = np.zeros((M,M))
outer = np.zeros((M,M))
for k in range(M):
    for i in range(M):
        for j in range(M):
            outer[i,j] = C[i,k]*C[j,k]
            
    P = P + outer*theta[k]
    
print(np.linalg.norm(P-P_pyscf, ord='fro'),'np.linalg.norm(P-P_pyscf)  should be zero...')
print(np.linalg.norm(P-P_pyscf_,ord='fro'),'np.linalg.norm(P-P_pyscf_) should be zero...')
##########################################################
# Build S^{(10x)} \in R^{(3\times M\times M)} (AO basis) 
# Note: anti symmetric
##########################################################
S_10 = nmr.rhf_dm.make_s10(mol)
print('np.shape(S^{(10)}) =', np.shape(S_10), M)

##########################################################
# Build H^{(10x)} \in R^{(3\times M\times M)} (AO basis)
# Note: anti symmetric
##########################################################
H_10 = nmr.rhf_dm.make_h10(mol,dm0=P)
print('np.shape(H^{(10)}) =', np.shape(H_10), M)

##########################################################
# Build H^{(10x)} = C^{\dagger}H^{(10x)}C_occ \in R^{(3\times M\times N)}
##########################################################
C_occ = C[:,theta>0]

# H^{(10x)}_{ij} = sum_{p,q}H_10_{xpq}C^{(10)}_{pi}C_occ_{qj}
H_10_ = lib.einsum('xpq,pi,qj->xij', H_10, C.conj(), C_occ)
#... or
H_10_ = lib.einsum('xpq,ip,qj->xij', H_10, C.T.conj(), C_occ)

# Note: to comply with C^{\dagger}H^{(1,0)}C_occ
# that is H^{(10)}_{ij} = sum_{p,q}C^{(10)\dagger}_{ip}H^{{(10x)_{pq}C_occ_{qj}
# this can be done:
H_10__ = lib.einsum('ip,xpq,qj->xij', C.T.conj(), H_10, C_occ)
test = H_10_ - H_10__
print(np.linalg.norm(test.reshape(-1, test.shape[-1]),ord='fro'),'should be zero...')


S_10_ = lib.einsum('xpq,pi,qj->xij', S_10, C.conj(), C_occ)
# same here...
S_10__= lib.einsum('ip,xpq,qj->xij', C.T.conj(), S_10, C_occ)
test = S_10_ - S_10__
print(np.linalg.norm(test.reshape(-1, test.shape[-1]),ord='fro'),'should be zero...')

##########################################################
# Below explicit build of H_10_
##########################################################
H_10_test = np.zeros((3,M,N))

## 1rst solution using np.matmul
H_10_test[0] = np.matmul(C.T.conj(),np.matmul(H_10[0],C_occ))
H_10_test[1] = np.matmul(C.T.conj(),np.matmul(H_10[1],C_occ))
H_10_test[2] = np.matmul(C.T.conj(),np.matmul(H_10[2],C_occ))
H_10_test = np.matmul(C.T.conj(), np.matmul(H_10, C_occ))
test = H_10_test - H_10_
print(np.linalg.norm(test.reshape(-1, test.shape[-1]),ord='fro'),'should be zero...')
## ...or
H_10_test = np.matmul(C.T.conj(),np.matmul(H_10,C_occ))
test = H_10_test - H_10_
print(np.linalg.norm(test.reshape(-1, test.shape[-1]),ord='fro'),'should be zero...')

## 2nd solution using explicit for loops
for i in range(M):
    for j in range(N):
        H_10_test[0,i,j] = 0.0
        H_10_test[1,i,j] = 0.0
        H_10_test[2,i,j] = 0.0
        for p in range(M):
            for q in range(M):
                H_10_test[0,i,j] = H_10_test[0,i,j] + C.T.conj()[i,p]*H_10[0,p,q]*C_occ[q,j]
                H_10_test[1,i,j] = H_10_test[1,i,j] + C.T.conj()[i,p]*H_10[1,p,q]*C_occ[q,j]
                H_10_test[2,i,j] = H_10_test[2,i,j] + C.T.conj()[i,p]*H_10[2,p,q]*C_occ[q,j]

# or in full 5 nested loops
for x in range(3):
    for i in range(M):
        for j in range(N):
            H_10_test[x,i,j] = 0.0
            for p in range(M):
                for q in range(M):
                    H_10_test[x,i,j] = H_10_test[x,i,j] + C.T.conj()[i,p]*H_10[x,p,q]*C_occ[q,j]

test = H_10_ - H_10_test
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(H_10_ - H_10_test) should be zero...')


##########################################################
# Get C^{10x) \in R^{(3\times M\times N)} (with_cphf=False => uncoupled)
##########################################################
C_10, C_e10 = nmr.rhf_dm.solve_mo1(mf,
                                   mo_energy=eigs,
                                   mo_coeff=C, 
                                   mo_occ=theta,
                                   h1=H_10_, 
                                   s1=S_10_, 
                                   with_cphf=False)

##########################################################
# Build D^{(10x)}_{oo} and D^{(10)}_vo \in R^{(3\times M\times M)} (with_cphf=False => uncoupled)
# Note: D^{(10x)}_{oo} = C_{occ}  X C_{occ}
# Note: D^{(10x)}_{vo} = C_{virt} X C_{occ}
# Note: X = X^{(10) = 
# with C_{occ} \in R^{(M\times  N)}
# with C_{virt}\in R^{(M\times (M-N))}
#      X in R^{(N\times N)}
##########################################################

occidx = theta > 0
viridx = theta == 0
C_occ  = C[:,occidx]
C_virt = C[:,viridx]
print('np.shape(C_10)  ', np.shape(C_10))
print('np.shape(C_occ) ', np.shape(C_occ))
print('np.shape(C_virt)', np.shape(C_virt))


## 1rst solution using pyscf code
# no double occupancy !
D_10_oo = np.asarray([reduce(np.dot, (C_occ,  X[occidx], C_occ.T.conj())) for X in C_10])
D_10_vo = np.asarray([reduce(np.dot, (C_virt, X[viridx], C_occ.T.conj())) for X in C_10])
print('np.shape(D_10_oo)', np.shape(D_10_oo))
print('np.shape(D_10_vo)', np.shape(D_10_vo))
print('D10_oo anti sym ?')
print(np.linalg.norm(D_10_oo[0] + D_10_oo.T[:,:,0],ord='fro'),'should be zero...')
print('D10_vo sym ?')
print(np.linalg.norm(D_10_vo[0] - D_10_vo.T[:,:,0],ord='fro'),'? ...')
print('D10_vo anti sym ?')
print(np.linalg.norm(D_10_vo[0] + D_10_vo.T[:,:,0],ord='fro'),'? ...')

## 2nd solution using matmul
D_10_oo_ = np.zeros((3,M,M))
D_10_vo_ = np.zeros((3,M,M))
for (X,i) in zip(C_10,range(3)):
    D_10_oo_[i] = np.matmul(C_occ, np.matmul(X[occidx],C_occ.T.conj()))    
    D_10_vo_[i] = np.matmul(C_virt,np.matmul(X[viridx],C_occ.T.conj()))

test = D_10_oo_ - D_10_oo
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(D_10_oo_ - D_10_oo) should be zero...')

test = D_10_vo_ - D_10_vo
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(D_10_vo_ - D_10_vo) should be zero...')

##########################################################
# Compute paramagnetic contribution to shielding tensor
##########################################################
shielding_nuc = range(mol.natm)
para_vir = np.zeros((len(shielding_nuc),3,3))
para_occ = np.zeros((len(shielding_nuc),3,3))
for n, atm_id in enumerate(shielding_nuc):
    mol.set_rinv_origin(mol.atom_coord(atm_id))
    ##########################################################
    # Compute H^{(01x)} \in R^{(3\times M\times M)} (AO basis)
    # Note: anti symmetric
    ##########################################################
    H_01 = mol.intor_asymmetric('int1e_prinvxp', 3)
    #print('np.shape(H^{(01)}) =', np.shape(H_01), M)
    ##########################################################
    # Compute 2\trace{D^{(01x)}_oo H^{(01y)}} + 2\trace{D^{(01x)}_vo H^{(01y)}}
    # as para_occ \in R^{natoms\times 3\times 3} + para_vir \in R^{natoms\times 3\times 3} 
    ##########################################################
    para_occ[n] = 2 * np.einsum('xji,yij->xy', D_10_oo, H_01) * 2 # *2 for occupation and *2 for + c.c.
    para_vir[n] = 2 * np.einsum('xji,yij->xy', D_10_vo, H_01) * 2 # *2 for occupation and *2 for + c.c.

msc_para = para_occ + para_vir    

# ... or
    
para_vir = np.zeros((len(shielding_nuc),3,3))
para_occ = np.zeros((len(shielding_nuc),3,3))
for n, atm_id in enumerate(shielding_nuc):
    mol.set_rinv_origin(mol.atom_coord(atm_id))
    ##########################################################
    # Compute H^{(10)} \in R^{(3\times M\times M)} (AO basis)
    # Note: anti symmetric
    ##########################################################
    H_01 = mol.intor_asymmetric('int1e_prinvxp', 3)
    for x in range(3):
        for y in range(3):
            para_occ[n,x,y] = 2 * np.trace(np.matmul(D_10_oo[x,:,:], H_01[y,:,:])) * 2
            para_vir[n,x,y] = 2 * np.trace(np.matmul(D_10_vo[x,:,:], H_01[y,:,:])) * 2 # *2 for occupation and *2 for + c.c.

msc_para_ = para_occ + para_vir

test = msc_para - msc_para_
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(msc_para - msc_para_) should be zero...')

# ... or

##########################################################
# Build D^{(01x)}_ov  = D^{(01x)\dagger}_vo 
##########################################################
D_10_ov = np.zeros((3,M,M))
for n, atm_id in enumerate(shielding_nuc):
    D_10_ov[n,:,:] = D_10_vo[n,:,:].T.conj()

##########################################################
# Build the full perturbed density matrix
##########################################################
D_10 = 2*D_10_oo + D_10_vo + D_10_vo

msc_para__ = np.zeros((len(shielding_nuc),3,3))
for n, atm_id in enumerate(shielding_nuc):
    mol.set_rinv_origin(mol.atom_coord(atm_id))
    ##########################################################
    # Compute H^{(10)} \in R^{(3\times M\times M)} (AO basis)
    # Note: anti symmetric
    ##########################################################
    H_01 = mol.intor_asymmetric('int1e_prinvxp', 3)
    ##########################################################
    # Build the full perturbed density matrix
    ##########################################################
    for x in range(3):
        for y in range(3):
            msc_para__[n,x,y] = 2 * np.trace(np.matmul(D_10[x,:,:],H_01[y,:,:]))

test = msc_para - msc_para__
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(msc_para - msc_para__) should be zero...')
                        
"""
##########################################################
# Redo a full response calculation from scratch and compare 
##########################################################

mf = scf.hf_dm.SCF(mol)
mf.conv_tol = 1e-14
mf.kernel()
m = nmr.RHF_DM(mf)
C_10_, C_e10_ = nmr.rhf_dm.solve_mo1(m,with_cphf=False)

#print(C_10)
#print(C_10_)
print(np.linalg.norm(C_10 - C_10_),'np.linalg.norm(C_10 - C_10_) should be zero...')

#m.kernel()
"""
##########################################################
# Redo a full response calculation from scratch
##########################################################
mf = scf.RHF(mol)
mf.conv_tol = 1e-14
mf.kernel()
mf_nmr = nmr.RHF_DM(mf)
mf_nmr.cphf = False
msc_dia = mf_nmr.dia()
msc_para_, para_vir_, para_occ_, dm10_oo_, dm10_vo_ = mf_nmr.para()

test = msc_para - msc_para_
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(msc_para - msc_para_ should be zero...')
 
#print(np.linalg.norm(D_10_oo*2 -  dm10_oo),'np.linalg.norm(D10_oo*2 -  dm10_oo) should be zero...')
#print(np.linalg.norm(D_10_vo*2 -  dm10_vo),'np.linalg.norm(D10_vo*2 -  dm10_vo) should be zero...')

#total = msc_dia + msc_para
#print(total*nist.ALPHA**2 * 1e6)
