#!/usr/bin/env python

# %%
'''

Computing NMR shielding constants
'''
import os

os.environ["PYSCF_EXT_PATH"] = "/Users/lioneltruflandier/pyscf-on/properties-on"

from pyscf import gto, dft, scf, lib
from pyscf.lib import logger
from pyscf.prop import nmr
from pyscf.data import nist
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from pyscf.scf.cphf import solve_withs1
from pyscf.lib import logger

mol = gto.M(atom='''
            C 0 0 0
            O 0 0 1.1747
            C 0 1 0
            ''',
            basis='sto-3G', verbose=20)

M = mol.nao_nr()
Ne= mol.tot_electrons()
N = int(mol.tot_electrons()/2) # only for RHF
print('### M (basis set size), Ne (nb. of electrons), N (nb. of occupied states)')
print(M,Ne,N)

mf = scf.RHF(mol)
conv, e_tot, eigs, C, theta = scf.hf_dm.kernel(mf,conv_tol=1e-8)
#conv, e_tot, eigs, C, theta = scf.hf.kernel(mf,conv_tol=1e-8)

print('###############################################################################')
print('# Build F^{(0)} \in R^{(M\times M)} (AO basis)')
print('###############################################################################')
mf.scf()
F = mf.get_fock()
print('np.shape(F^{(0)}) =', np.shape(F), M)

print('###############################################################################')
print('C \in R^{(M\times M)} = unperturbed matrix of MO coeff.')
print('###############################################################################')
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
H_10 = nmr.rhf_dm.make_h10(mol,dm0=P_pyscf)
print('np.shape(H^{(10)}) =', np.shape(H_10), M)

##########################################################
# Build S^{(0)} \in R^{(M\times M)} (AO basis)
##########################################################
S = mol.intor("int1e_ovlp")
print('np.shape(S^{(0)}) =', np.shape(S), M)


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

for x in range(3):
    print(np.linalg.norm(S_10[x,:,:] - S_10[x,:,:].T,ord='fro'),'S - S.T ?...')
    print(np.linalg.norm(H_10[x,:,:] - H_10[x,:,:].T,ord='fro'),'H - H.T ?...')

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
with_cpscf = True

mf_nmr = nmr.RHF_DM(mf)
C_10, C_e10 = nmr.rhf_dm.solve_mo1(mf_nmr,
                                   mo_energy=eigs,
                                   mo_coeff=C, 
                                   mo_occ=theta,
                                   h1=H_10_, 
                                   s1=S_10_, 
                                   with_cphf=with_cpscf)



#print(np.shape(C_10[0]) )
with np.printoptions(precision=10, suppress=True, formatter={'float': '{:12.4f}'.format}, linewidth=200):
    print(C_10[0])

C_x_pad = np.pad(C_10[0], [(0, 0), (0, M-N)], mode='constant', constant_values=0)
C_y_pad = np.pad(C_10[1], [(0, 0), (0, M-N)], mode='constant', constant_values=0)
C_z_pad = np.pad(C_10[2], [(0, 0), (0, M-N)], mode='constant', constant_values=0)

C_10_   = np.array([C_x_pad,C_y_pad,C_z_pad]) 
C_10_t  = np.array([C_x_pad.T.conj(),C_y_pad.T.conj(),C_z_pad.T.conj()]) 

S_x_pad = np.pad(S_10_[0], [(0, 0), (0, M-N)], mode='constant', constant_values=0)
S_y_pad = np.pad(S_10_[1], [(0, 0), (0, M-N)], mode='constant', constant_values=0)
S_z_pad = np.pad(S_10_[2], [(0, 0), (0, M-N)], mode='constant', constant_values=0)

S_10_pp   = np.array([S_x_pad,S_y_pad,S_z_pad]) 
S_10_ppt  = np.array([S_x_pad.T.conj(),S_y_pad.T.conj(),S_z_pad.T.conj()]) 


np.set_printoptions(precision=2, suppress=True)
C_10_C_10d = C_10_   + C_10_t.conj()
S_10_S_10d = S_10_pp + S_10_ppt.conj()


#C_10_C_10d_AO = C @ C_10_C_10d @ C.T.conj() 
print('#######################################@')
with np.printoptions(precision=10, suppress=True, formatter={'float': '{:12.4f}'.format}, linewidth=100):
    print(np.shape(C_10[0]))
    print(np.shape(S_10[0]))

    print(C_x_pad)
print('#######################################@')

#print(C_10_C_10d_AO[0])
#print(C_10_C_10d[0])
#print(S_10_S_10d[0])
#print(S_10_S_10d[0] + 2*C_10_C_10d[0])
#print(S_10_S_10d[0] + C_10_[0])

#print(S_10_[0] + C_10[0])

#print(C_10_C_10d_AO[0] - S_10[0])

#test = C_10_C_10d + S_10
#print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(C_10.T + C_10 + S_10) should be zero...')

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
C_vir  = C[:,viridx]
print('np.shape(C_10) ', np.shape(C_10))
print('np.shape(C_occ)', np.shape(C_occ))
print('np.shape(C_vir)', np.shape(C_vir))


## 1rst solution using pyscf code
# no double occupancy !
D_10_oo = np.asarray([reduce(np.dot, (C_occ, X[occidx], C_occ.T.conj())) for X in C_10])
D_10_vo = np.asarray([reduce(np.dot, (C_vir, X[viridx], C_occ.T.conj())) for X in C_10])
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
    print(np.shape(X),np.shape(X[occidx]),np.shape(X[viridx]))    
    D_10_oo_[i] = np.matmul(C_occ,np.matmul(X[occidx],C_occ.T.conj()))    
    D_10_vo_[i] = np.matmul(C_vir,np.matmul(X[viridx],C_occ.T.conj()))


test = D_10_oo_ - D_10_oo
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(D_10_oo_ - D_10_oo) should be zero...')

test = D_10_vo_ - D_10_vo
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(D_10_vo_ - D_10_vo) should be zero...')

D_10__ = np.zeros((3,M,M))
for i in range(3):
    D_10__[i] =  D_10_oo_[i] + D_10_vo_[i]
    D_10__[i] += D_10__[i].T.conj() 

#print(D_10_vo_[0] - D_10_vo_[0].T)
#%%
D_10_ = np.zeros((3,M,M),dtype=complex)
for (X,i) in zip(C_10,range(3)):
    D_10_[i]  = np.matmul(np.matmul(C,X),C_occ.T.conj())*1j
    D_10_[i] += D_10_[i].T.conj()

D_10_ =  np.imag(D_10_)
 
print('############### D_10 from U')
print('############### D_10 from U')
# %%
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
    # Compute H^{(01x)} \in R^{(3\times M\times M)} (AO basis)
    # Take care: anti symmetric and pure imaginary !
    ##########################################################
    H_01 = mol.intor_asymmetric('int1e_prinvxp', 3)
    for x in range(3):
        for y in range(3):
            para_occ[n,x,y] = 2 * np.trace(np.matmul(D_10_oo[x,:,:], H_01[y,:,:])) * 2
            para_vir[n,x,y] = 2 * np.trace(np.matmul(D_10_vo[x,:,:], H_01[y,:,:])) * 2 # *2 for occupation and *2 for + c.c.

msc_para_ = para_occ + para_vir

test = msc_para - msc_para_
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(msc_para - msc_para_) should be zero...')


print(np.matmul(D_10_vo[0,:,:],H_01[0,:,:]))
# ... or

##########################################################
# Build D^{(01x)}_ov 
# = -D^{(01x)\dagger}_vo because H^{(01x)} is pure imaginary !
##########################################################
D_10_ov = np.zeros((3,M,M))
for n, atm_id in enumerate(shielding_nuc):
    D_10_ov[n,:,:] = -D_10_vo[n,:,:].T.conj()

##########################################################
# Build the full perturbed density matrix
# Take care: D^{(01x)
##########################################################
D_10 = 2*D_10_oo + D_10_ov + D_10_vo

msc_para__ = np.zeros((len(shielding_nuc),3,3))
for n, atm_id in enumerate(shielding_nuc):
    mol.set_rinv_origin(mol.atom_coord(atm_id))
    ##########################################################
    # Compute H^{(01x)} \in R^{(3\times M\times M)} (AO basis)
    # Take care: anti symmetric and pure imaginary !
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
##########################################################
# Compute the density matrix from Dodds an McWeeny paper 
# eq. 29 in MOLECULAR PHYSICS, 1977, VOL. 34, No. 6, 1779
##########################################################


# Build D = D^{(0)} or get it from pyscf
D = np.zeros((M,M))
outer = np.zeros((M,M))
for k in range(N):
    outer = np.outer(C[:,k],C[:,k])            
    D = D + outer   
    
test = P/2 - D
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(D - D_) should be zero...')

##########################################################
# Build D_10_oo_mcw = - D^{(0)}S^{(10)}D^{(0)}
##########################################################
D_10_oo_mcw = np.zeros((3,M,M))
for x in range(3):
    D_10_oo_mcw[x] = - np.matmul(D,np.matmul(S_10[x],D)) 

#print(occidx)
#print(viridx)
N_o = list(occidx).count(True)
N_v = list(viridx).count(True)
e_v = eigs[viridx]
e_o = eigs[occidx]

##########################################################
# Build D_10_ov_mcw = ...
##########################################################

D_10_ov_mcw = np.zeros((3,M,M))
for x in range(3):
    for k in range(N_o):
        for l in range(N_v):
            outer  = np.outer( C_occ[:,k],C_vir[:,l]) 
            scalar = np.inner( C_occ[:,k], np.matmul(H_10[x] - e_o[k]*S_10[x], C_vir[:,l]) )/(e_o[k] - e_v[l])
            D_10_ov_mcw[x] = D_10_ov_mcw[x] + scalar*outer
            
##########################################################
# Get D_10_vv_mcw = ...
##########################################################
D_10_vo_mcw = np.zeros((3,M,M))
for n, atm_id in enumerate(shielding_nuc):
    D_10_vo_mcw[n,:,:] = - D_10_ov_mcw[n,:,:].T.conj()

##########################################################
# eq. 29 in MOLECULAR PHYSICS, 1977, VOL. 34, No. 6, 1779
# still this 2 factor ; I think I know why
##########################################################
D_10_mcw = D_10_oo_mcw + D_10_ov_mcw + D_10_vo_mcw

#test = D_10_ov_mcw - D_10_ov
#print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(D_10_ov_mcw - D_10_ov) should be zero...')        

#test = D_10_vo_mcw - D_10_vo
#print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(D_10_vo_mcw - D_10_vo) should be zero...')        

test = D_10_oo_mcw - 2*D_10_oo
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(D_10_oo_mcw - D_10_oo) should be zero...')        

test = D_10_mcw - D_10
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(D_10_mcw - D_10) should be zero...')        

test = D_10_ov_mcw - D_10_ov
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(D_10_ov_mcw - D_10_ov) should be zero...')        

#%%
occidx = theta > 0
orbo = C[:,occidx]
nocc = orbo.shape[1]
nao, nmo = C.shape
dm1 = [reduce(np.dot, (C, x*2, orbo.T.conj()))
       for x in C_10.reshape(-1,nmo,nocc)]

dm1 = np.asarray([d1-d1.conj().T for d1 in dm1])

test = dm1 - 2*D_10

#plt.figure('dm1')
#plt.matshow(dm1[1])
#plt.colorbar()
#plt.figure('D10')
#plt.matshow(2*D_10[1]-dm1[1])
#plt.colorbar()
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(dm1 - D_10) should be zero...')        

#%%
fvind = nmr.rhf_dm.gen_vind(mf, mo_coeff=C, mo_occ=theta)

print('separate call of fvind')
G_10 = fvind(mo1=C_10)
print('np.shape(G_10) =', np.shape(G_10), M)

fvind_dm = nmr.rhf_dm.gen_vind_dm(mf, mo_coeff=C, mo_occ=theta)
G_10_dm = fvind_dm(dm1=dm1)
print('np.shape(G_10) =', np.shape(G_10_dm), M)

test = G_10 - G_10_dm
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(G_10 - G_10_dm) should be zero...')      
#%%

D_10_ov_mcw_cpscf = np.zeros((3,M,M))

if ( with_cpscf ):
    fvind_ao = nmr.rhf_dm.gen_vind_ao(mf, mo_coeff=C, mo_occ=theta)   
    G_10 = fvind_ao(dm1=2*D_10)
    
    for x in range(3):
        for k in range(N_o):
            for l in range(N_v):
                outer  = np.outer( C_occ[:,k],C_vir[:,l]) 
                scalar = np.inner( C_occ[:,k], np.matmul(H_10[x] + G_10[x] - e_o[k]*S_10[x], C_vir[:,l]) )/(e_o[k] - e_v[l])
                D_10_ov_mcw_cpscf[x] = D_10_ov_mcw_cpscf[x] + scalar*outer

    D_10_vo_mcw_cpscf = np.zeros((3,M,M))
    for n, atm_id in enumerate(shielding_nuc):
        D_10_vo_mcw_cpscf[n,:,:] = - D_10_ov_mcw_cpscf[n,:,:].T.conj()

    D_10_mcw_cpscf = D_10_oo_mcw + D_10_ov_mcw_cpscf + D_10_vo_mcw_cpscf

test = D_10_mcw_cpscf - D_10
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(D_10_mcw_cpscf - D_10) should be zero...')        

test =  D_10_  - D_10
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(D_10_ - D_10) should be zero...')        

test = D_10_ov_mcw_cpscf - D_10_ov
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(D_10_ov_mcw_cpscf - D_10_ov) should be zero...')        


#%%


fvind = nmr.rhf_dm.gen_vind(mf, mo_coeff=C, mo_occ=theta)
mo_occ = theta
mo_energy = eigs
h1 = lib.einsum('xpq,pi,qj->xij', H_10, C.conj(), C_occ)
s1 = lib.einsum('xpq,pi,qj->xij', S_10, C.conj(), C_occ)
print(np.shape(S_10))
mo1, mo_e1 = solve_withs1(fvind, mo_energy=mo_energy, mo_occ=mo_occ, h1=h1, s1=s1,
             max_cycle=50, tol=1e-9, hermi=False, verbose=logger.WARN,
             level_shift=0)


test = mo1 - C_10
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(mo1 - C_10) should be zero...')        

#%%


def solve_withs1_(fvind, mo_energy, mo_occ, h1, s1,
                 max_cycle=50, tol=1e-9, hermi=False, verbose=logger.WARN,
                 level_shift=0):
    '''For field dependent basis. First order overlap matrix is non-zero.
    The first order orbitals are set to
    C^1_{ij} = -1/2 S1
    e1 = h1 - s1*e0 + (e0_j-e0_i)*c1 + vhf[c1]

    Kwargs:
        level_shift : float
            Add to diagonal terms to slightly improve the convergence speed of
            Krylov solver

    Returns:
        First order orbital coefficients (in MO basis) and first order orbital
        energy matrix
    '''
    print('solve_withs1')
    assert not hermi
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    occidx = mo_occ > 0
    print('occidx', occidx)
    viridx = mo_occ == 0
    print('viridx', viridx)

    # e_a in R^{virt}
    e_a = mo_energy[viridx]
    print('e_a =', e_a)
    # e_i = R^{occ}
    e_i = mo_energy[occidx]
    # e_ai = e_a e_i^t in R^{occ x virt}
    print('e_i =', e_i)
    e_ai = 1 / (e_a[:,None] + level_shift - e_i)
    print('np.shape(e_ai)',np.shape(e_ai))
    print('e_ai =')
    print(e_ai)
    
    nvir, nocc = e_ai.shape
    nmo = nocc + nvir

    print('######### solve_withs1 ####################################')
    print('# solve_withs1')
    print('######### solve_withs1 ####################################')
    print('s1')
    print(s1)

    print('np.shape(s1),nmo,nocc',np.shape(s1),nmo,nocc)
    s1 = s1.reshape(-1,nmo,nocc)
    print('np.shape(s1),nmo,nocc',np.shape(s1),nmo,nocc)

    print('s1')
    print(s1)

    print('np.shape(h1),nmo,nocc',np.shape(h1),nmo,nocc)
    #hs = C^{\dagger} H1 C_occ - C^{\dagger} S1 C_occ*e_i
    hs = mo1base = h1.reshape(-1,nmo,nocc) - s1*e_i
    print('np.shape(hs),nmo,nocc',np.shape(hs),nmo,nocc)

    mo1base = hs.copy()
    print('np.shape(mo1base),nmo,nocc',np.shape(mo1base),nmo,nocc)
    print('np.shape(mo1base[:,viridx]),nmo,nocc',np.shape(mo1base[:,viridx]),nmo,nocc)
    
    #hs_ai = -(C^{\dagger}_virt_a H1 C_occ_i - C^{\dagger}_a S1 C_occ_i*e_i)/(e_a - e_i)
    mo1base[:,viridx,:] *= -e_ai   
    print('mo1base')
    print(mo1base[0])
    #hs_ii = -(C^{\dagger}_occ_i S1 C_occ_i 
    mo1base[:,occidx,:] = -s1[:,occidx,:] * .5
    print('np.shape(mo1base[:,occidx]),nmo,nocc',np.shape(mo1base[:,occidx]),nmo,nocc)
    print(np.shape(mo1base[:,occidx]),nmo,nocc)
    print('mo1base')
    print(mo1base[0])
    
    print('s1[:,occidx]')
    print(-s1[:,occidx] * .5)
    print('s1')
    print(-s1 * .5)

    def vind_vo(mo1):
        mo1 = mo1.reshape(-1, nmo, nocc)
        v = fvind(mo1).reshape(-1, nmo, nocc)
        if level_shift != 0:
            v -= mo1 * level_shift
        v[:,viridx,:] *= e_ai
        v[:,occidx,:] = 0
        return v.reshape(-1, nmo*nocc)
    
    print('######### lib.krylov in  ####################################')
    mo1 = lib.krylov(vind_vo, mo1base.reshape(-1, nmo*nocc),
                     tol=tol, max_cycle=max_cycle, hermi=hermi, verbose=log)
    print('######### lib.krylov out ####################################')
    
    mo1 = mo1.reshape(-1, nmo, nocc)
    print('mo1[:,occidx]')
    print(mo1[:,occidx])

    mo1[:,occidx] = mo1base[:,occidx]
    print(mo1[:,occidx])

    log.timer('krylov solver in CPHF', *t0)

    hs += fvind(mo1).reshape(-1, nmo, nocc)
    mo1[:,viridx] = hs[:,viridx] / (e_i - e_a[:,None])

    # mo_e1 has the same symmetry as the first order Fock matrix (hermitian or
    # anti-hermitian). mo_e1 = v1mo - s1*lib.direct_sum('i+j->ij',e_i,e_i)
    mo_e1 = hs[:,occidx,:]
    mo_e1 += mo1[:,occidx] * (e_i[:,None] - e_i)

    if h1.ndim == 3:
        return mo1, mo_e1
    else:
        assert h1.ndim == 2
        return mo1[0], mo_e1[0]

print('s1 input')
print(s1)


# s1 = C^{\dagger} S1 C_occ
# h1 = C^{\dagger} S1 C_occ

mo1_, mo_e1_ = solve_withs1_(fvind, mo_energy=mo_energy, mo_occ=mo_occ, h1=h1, s1=s1,
             max_cycle=50, tol=1e-9, hermi=False, verbose=logger.WARN,
             level_shift=0)


test = mo1_ - C_10
print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(mo1_ - C_10) should be zero...')        



A = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(A)
print(A[:,[True,False,True]])
#%%
#e_a = eigs[viridx]
#print('### e_a: 1D e_a(i) size (M-N)', np.shape(e_a))
#print(e_a)

#e_i = eigs[occidx]
#print('### e_i: 1D e_i(i) size N', np.shape(e_i))
#print(e_i)

#e_ai = (e_a.reshape(-1,1) - e_i)
#print('### e_ai: 2D 1/(e_a(i) - (e_i(j))) size (M-N)xN',np.shape(e_ai))
 
#R = (np.identity(M) - 2 * S @ P) @ F
#eig, U = np.linalg.eig(R)
#print(np.sort(eig),np.linalg.cond(R),np.linalg.cond(np.linalg.inv(S)))
#fig = plt.figure()
#mat = plt.matshow(R)

##########################################################
# Redo a full response calculation from scratch and compare 
##########################################################

#mf = scf.hf_dm.SCF(mol)
#mf.conv_tol = 1e-14
#mf.kernel()
#m = nmr.RHF_DM(mf)
#C_10_, C_e10_ = nmr.rhf_dm.solve_mo1(m,with_cphf=False)

#print(C_10)
#print(C_10_)
#print(np.linalg.norm(C_10 - C_10_),'np.linalg.norm(C_10 - C_10_) should be zero...')

#m.kernel()


##########################################################
# Redo a full response calculation from scratch
##########################################################

#mf = scf.RHF_DM(mol)
#mf.conv_tol = 1e-14
#mf.kernel()
#mf_nmr = nmr.RHF_DM(mf)
#mf_nmr.cphf = True
#mf_nmr.shielding()

#msc_para_, para_vir_, para_occ_, dm10_oo_, dm10_vo_ = mf_nmr.para()
#msc_para_, para_vir_, para_occ_= mf_nmr.para()

#test = msc_para - msc_para_
#print(np.linalg.norm( test.reshape(-1, test.shape[-1]) ,ord='fro'),'np.linalg.norm(msc_para - msc_para_ should be zero...')
 
#print(np.linalg.norm(D_10_oo*2 -  dm10_oo),'np.linalg.norm(D10_oo*2 -  dm10_oo) should be zero...')
#print(np.linalg.norm(D_10_vo*2 -  dm10_vo),'np.linalg.norm(D10_vo*2 -  dm10_vo) should be zero...')

#total = msc_dia + msc_para
#print(total*nist.ALPHA**2 * 1e6)
#mf_nmr.shielding()


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