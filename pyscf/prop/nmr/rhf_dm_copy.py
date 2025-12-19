#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic NMR shielding tensor
'''


'''
### LAT: Identification of terms wrt. to Wolinski J. Am. Chem. Soc., Vol. 112, No. 23, 1990
'''


from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.scf import cphf
from pyscf.scf import _response_functions  # noqa
from pyscf.data import nist
from scipy import linalg
from scipy.sparse.linalg  import cg, gmres
from scipy.sparse import csc_matrix
from sksparse.cholmod import cholesky
from pyscf.prop.nmr.utils import inv, diis, build_ab, build_q, gershgorin_max, gershgorin_min
from pyscf.prop.nmr.huckel import get_system, dummy_F1_S1, overlap
import time
import matplotlib.pyplot as plt

def dia(nmrobj, gauge_orig=None, shielding_nuc=None, dm0=None):
    '''Diamagnetic part of NMR shielding tensors.

    See also J. Olsen et al., Theor. Chem. Acc., 90, 421 (1995)
    '''
    if shielding_nuc is None: shielding_nuc = nmrobj.shielding_nuc
    if dm0 is None: dm0 = nmrobj._scf.make_rdm1()

    mol = nmrobj.mol
    mf = nmrobj._scf

    if getattr(mf, 'with_x2c', None):
        raise NotImplementedError('X2C for NMR shielding')

    if getattr(mf, 'with_qmmm', None):
        raise NotImplementedError('NMR shielding with QM/MM')

    if getattr(mf, 'with_solvent', None):
        raise NotImplementedError('NMR shielding with Solvent')

    if gauge_orig is not None:
        # Note the side effects of set_common_origin
        mol.set_common_origin(gauge_orig)

    msc_dia = []
    for n, atm_id in enumerate(shielding_nuc):
        with mol.with_rinv_origin(mol.atom_coord(atm_id)):
            # a11part = (B dot) -1/2 frac{\vec{r}_N}{r_N^3} r (dot mu)
            if gauge_orig is None:
                h11 = mol.intor('int1e_giao_a11part', comp=9)
            else:
                h11 = mol.intor('int1e_cg_a11part', comp=9)
            e11 = numpy.einsum('xij,ij->x', h11, dm0).reshape(3,3)
            e11 = e11 - numpy.eye(3) * e11.trace()
            if gauge_orig is None:
                h11 = mol.intor('int1e_a01gp', comp=9)
                e11 += numpy.einsum('xij,ij->x', h11, dm0).reshape(3,3)
        msc_dia.append(e11)
    return numpy.array(msc_dia).reshape(-1, 3, 3)

def para(nmrobj, mo10=None, mo_coeff=None, mo_occ=None, shielding_nuc=None):
    '''Paramagnetic part of NMR shielding tensors.
    '''

    if mo_coeff is None:      mo_coeff = nmrobj._scf.mo_coeff
    if mo_occ is None:        mo_occ = nmrobj._scf.mo_occ
    if shielding_nuc is None: shielding_nuc = nmrobj.shielding_nuc
    if mo10 is None: mo10 = nmrobj.solve_mo1()[0]

    mol = nmrobj.mol
    para_vir = numpy.empty((len(shielding_nuc),3,3))
    para_occ = numpy.empty((len(shielding_nuc),3,3))
    occidx = mo_occ > 0
    viridx = mo_occ == 0
    orbo = mo_coeff[:,occidx]
    orbv = mo_coeff[:,viridx]
    # *2 for double occupancy
    dm10_oo = numpy.asarray([reduce(numpy.dot, (orbo, x[occidx]*2, orbo.T.conj())) for x in mo10])
    dm10_vo = numpy.asarray([reduce(numpy.dot, (orbv, x[viridx]*2, orbo.T.conj())) for x in mo10])
    for n, atm_id in enumerate(shielding_nuc):
        mol.set_rinv_origin(mol.atom_coord(atm_id))
        # H^{01} = 1/2(A01 dot p + p dot A01) => (a01p + c.c.)/2 ~ <a01p>
        # Im[A01 dot p] = Im[vec{r}/r^3 x vec{p}] = Im[-i p (1/r) x p] = -p (1/r) x p
        h01i = mol.intor_asymmetric('int1e_prinvxp', 3)  # = -Im[H^{01}]
        # <H^{01},MO^1> = - Tr(Im[H^{01}],Im[MO^1]) = Tr(-Im[H^{01}],Im[MO^1])
        para_occ[n] = numpy.einsum('xji,yij->xy', dm10_oo, h01i) * 2 # *2 for + c.c.
        para_vir[n] = numpy.einsum('xji,yij->xy', dm10_vo, h01i) * 2 # *2 for + c.c.
    msc_para = para_occ + para_vir

    return msc_para, para_vir, para_occ

def para_dm(nmrobj, method = 'mcw', dm10 = None, mo_coeff=None, mo_occ=None, shielding_nuc=None):
    '''Paramagnetic part of NMR shielding tensors, using DM approach in AO basis
    '''

    if mo_coeff is None: mo_coeff = nmrobj._scf.mo_coeff
    if mo_occ is None: mo_occ = nmrobj._scf.mo_occ
    if shielding_nuc is None: shielding_nuc = nmrobj.shielding_nuc

    mol = nmrobj.mol
        
    if ( method == 'mcw'  ): 
        if dm10 is None: dm10 = solve_dm10_mcweeny(nmrobj, mo_coeff, mo_occ)[0]
        
    elif ( method == 'slv' ): 
        if dm10 is None: dm10 = solve_dm10_sylvester(nmrobj, mo_coeff, mo_occ)
        
    elif ( method == 'tc2' ):
        if dm10 is None: dm10 = purification_first_order(nmrobj)[1]
        
    elif ( method == 'hpcp' ):
        if dm10 is None: dm10 = purification_first_order(nmrobj, method='hpcp')[1]
        
    else:
         raise ValueError(f"Unknown method '{method}'. Choose from: 'mcw', 'slv', 'tc2', 'hpcp'.")
    
    msc_para = numpy.zeros((len(shielding_nuc),3,3))
    for n, atm_id in enumerate(shielding_nuc):
        mol.set_rinv_origin(mol.atom_coord(atm_id))
        h01i = mol.intor_asymmetric('int1e_prinvxp', 3)
        for x in range(3):
            for y in range(3):
                msc_para[n,x,y] = 2 * numpy.trace(dm10[x,:,:] @ h01i[y,:,:])       
                    
    return msc_para

def make_h10(mol, dm0, gauge_orig=None, verbose=logger.WARN):
    '''Imaginary part of first order Fock operator

    Note the side effects of set_common_origin
    '''
    log = logger.new_logger(mol, verbose)
    if gauge_orig is None:
        # A10_i dot p + p dot A10_i consistents with <p^2 g>
        # A10_j dot p + p dot A10_j consistents with <g p^2>
        # 1/2(A10_j dot p + p dot A10_j) => Im[1/4 (rjxp - pxrj)] = -1/2 <irjxp>
        log.debug('First-order GIAO Fock matrix')
        h1 = -.5 * mol.intor('int1e_giao_irjxp', 3) + make_h10giao(mol, dm0)
    else:
        with mol.with_common_origin(gauge_orig):
            h1 = -.5 * mol.intor('int1e_cg_irxp', 3)
    return h1

def get_jk(mol, dm0):
    # J = Im[(i i|\mu g\nu) + (i gi|\mu \nu)] = -i (i i|\mu g\nu)
    # K = Im[(\mu gi|i \nu) + (\mu i|i g\nu)]
    #   = [-i (\mu g i|i \nu)] - h.c.   (-h.c. for anti-symm because of the factor -i)
    intor = mol._add_suffix('int2e_ig1')
    vj, vk = _vhf.direct_mapdm(intor,  # (g i,j|k,l)
                               'a4ij', ('lk->s1ij', 'jk->s1il'),
                               dm0, 3, # xyz, 3 components
                               mol._atm, mol._bas, mol._env)
    vk = vk - numpy.swapaxes(vk, -1, -2)
    return -vj, -vk

def make_h10giao(mol, dm0):
    vj, vk = get_jk(mol, dm0)
    h1 = vj - .5 * vk
    # Im[<g\mu|H|g\nu>] = -i * (gnuc + gkin)
    h1 -= mol.intor_asymmetric('int1e_ignuc', 3)
    if mol.has_ecp():
        h1 -= mol.intor_asymmetric('ECPscalar_ignuc', 3)
    h1 -= mol.intor('int1e_igkin', 3)
    return h1

def make_s10(mol, gauge_orig=None):
    '''First order overlap matrix wrt external magnetic field.'''

    if gauge_orig is None:
        # Im[<g\mu |g\nu>]
        s1 = -mol.intor_asymmetric('int1e_igovlp', 3)
    else:
        nao = mol.nao_nr()
        s1 = numpy.zeros((3,nao,nao))
    return s1
get_ovlp = make_s10

def _solve_mo1_uncoupled(mo_energy, mo_occ, h1, s1):
    '''uncoupled first order equation'''

    e_a = mo_energy[mo_occ==0]
    e_i = mo_energy[mo_occ>0]
    e_ai = 1 / (e_a.reshape(-1,1) - e_i)

    hs = h1 - s1 * e_i

    mo10 = numpy.empty_like(hs)
    mo10[:,mo_occ==0,:] = -hs[:,mo_occ==0,:] * e_ai
    mo10[:,mo_occ>0,:] = -s1[:,mo_occ>0,:] * .5

    e_ji = e_i.reshape(-1,1) - e_i
    mo_e10 = hs[:,mo_occ>0,:] + mo10[:,mo_occ>0,:] * e_ji

    return mo10, mo_e10

#TODO: merge to hessian.rhf.solve_mo1 function
def solve_mo1(nmrobj, mo_energy=None, mo_coeff=None, mo_occ=None,
              h1=None, s1=None, with_cphf=None):
    '''Solve the first order equation

    Kwargs:
        with_cphf : boolean or  function(dm_mo) => v1_mo
            If a boolean value is given, the value determines whether CPHF
            equation will be solved or not. The induced potential will be
            generated by the function gen_vind.
            If a function is given, CPHF equation will be solved, and the
            given function is used to compute induced potential
    '''
    t0 = time.perf_counter()
    
    if mo_energy is None: mo_energy = nmrobj._scf.mo_energy
    if mo_coeff is None: mo_coeff = nmrobj._scf.mo_coeff
    if mo_occ is None: mo_occ = nmrobj._scf.mo_occ
    if with_cphf is None: with_cphf = nmrobj.cphf

    cput1 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(nmrobj.stdout, nmrobj.verbose)

    mol = nmrobj.mol
    orbo = mo_coeff[:,mo_occ>0]
    if h1 is None:
        ### Compute (C_i+C_a)F^(1)C_i with C_i (M*M)
        ### with C_i (M*N) columns vector of unperturbed occ.
        ### with C_a (M*(M-N)) columns vector of unperturbed unocc.
        ### with F^(1) (3*M*M) perturbed Fock matrix in 3 directions
        ### h1 = F^{(1)}
        dm0 = nmrobj._scf.make_rdm1(mo_coeff, mo_occ)
        h1 = lib.einsum('xpq,pi,qj->xij', nmrobj.get_fock(dm0),
                        mo_coeff.conj(), orbo)
        cput1 = log.timer('first order Fock matrix', *cput1)
        
    if s1 is None:
        ### Compute (C_i+C_a)S^(1)C_i with C_i (M*M)
        ### with C_i (M*N) columns vector of occ.
        ### with C_a (M*(M-N)) columns vector of unocc.
        ### with S^(1) (3*M*M) perturbed overlap matrix in 3 directions
        s1 = lib.einsum('xpq,pi,qj->xij', nmrobj.get_ovlp(mol),
                        mo_coeff.conj(), orbo)

    if with_cphf:
        if callable(with_cphf):
            vind = with_cphf
        else:
            vind = gen_vind(nmrobj._scf, mo_coeff, mo_occ)
        mo10, mo_e10 = cphf.solve(vind, mo_energy, mo_occ, h1, s1,
                                  nmrobj.max_cycle_cphf, nmrobj.conv_tol,
                                  verbose=log)
    else:
        mo10, mo_e10 = _solve_mo1_uncoupled(mo_energy, mo_occ, h1, s1)

    log.timer('solving mo1 eqn', *cput1)
    
    t1 = time.perf_counter()
    total_time = t1 - t0
    print(f"time spent in M0 routine: {total_time}")
    
    return mo10, mo_e10

def _solve_dm10_uncoupled(mo_coeff, mo_occ, mo_energy, h1, s1, D0):
   
    occidx = mo_occ > 0
    viridx = mo_occ == 0
    nocc, nvir = numpy.sum(occidx), numpy.sum(viridx)
    eo, ev = mo_energy[occidx], mo_energy[viridx]
    Co, Cv = mo_coeff[:, occidx], mo_coeff[:, viridx]
    cart, M = s1.shape[0], s1.shape[1]
    #D0 = Co @ Co.conj().T
    Doo = -.5* (D0 @ s1) @ D0
    Dov = numpy.zeros((cart, M, M))
    
    
    for n in range(cart):
        for i in range(nocc):
            nominator = h1[n]-eo[i]*s1[n]
            for j in range(nvir):
                Xov = numpy.outer(Co[:,i],Cv[:,j]) 
                denominator = eo[i] - ev[j]                 
                alpha = numpy.inner(Co[:,i],(( nominator / denominator) @ Cv[:,j]))                           
                Dov[n] += alpha * Xov                        
                
    Dvo = - Dov.transpose(0,2,1)
    D1 = 2*Doo + Dov + Dvo
    
    return D1, Doo, Dov, Dvo

def solve_dm10_mcweeny(nmrobj, mo_coeff=None, mo_occ=None, mo_energy=None,
               h1=None, s1=None, with_cphf=None):
    '''Build first-order density matrix in AO basis
        based on SOS-McWeeny-DMPT approach'''
     
    t0 = time.perf_counter()
     
    # if mo_coeff is None: mo_coeff = nmrobj._scf.mo_coeff
    # if mo_occ is None: mo_occ = nmrobj._scf.mo_occ
    # if mo_energy is None: mo_energy = nmrobj._scf.mo_energy
    # if with_cphf is None: with_cphf = nmrobj.cphf
    # dm0 = nmrobj._scf.make_rdm1(mo_coeff, mo_occ)
    # mol = nmrobj.mol
    # if h1 is None:
    #     h1 = make_h10(mol, dm0, gauge_orig=nmrobj.gauge_orig)
    # if s1 is None:
    #     s1 = make_s10(mol, gauge_orig=nmrobj.gauge_orig)
    
    F0, S0, D0, N = get_system(1000, 1000)
    h1, s1 = dummy_F1_S1(F0)
    #Sinv = numpy.linalg.inv(S0)
    mo_energy, mo_coeff = np.linalg.eigh(F0)
    nocc = N
    nvir = 1000 - N
    #mo_occ = np.array([2,2,2,2,2,0,0,0,0,0])
    mo_occ = np.zeros(1000)
    mo_occ[:nocc] = 2.0
    
    if not with_cphf:
        D1, Doo, Dov, Dvo = _solve_dm10_uncoupled(mo_coeff, mo_occ,
                                                  mo_energy, h1, s1, D0)
        t1 = time.perf_counter()
        total_time = t1 - t0
        print(f"time spent in DM routine: {total_time}")
        return D1, Doo, Dov, Dvo

    else: 
        occidx = mo_occ > 0
        viridx = mo_occ == 0
        nocc, nvir = numpy.sum(occidx), numpy.sum(viridx)
        eo, ev = mo_energy[occidx], mo_energy[viridx]
        Co, Cv = mo_coeff[:, occidx], mo_coeff[:, viridx]
        cart, M = s1.shape[0], s1.shape[1]
        #D0 = .5*dm0
        #S0 = mol.intor('int1e_ovlp')
        #F0 = nmrobj._scf.get_fock()
        Doo = -.5 * (D0 @ s1) @ D0
        
        vresp = nmrobj._scf.gen_response(singlet=True, hermi=2)
        
        max_cycle = 2000
        conv_tol = 1e-8
        D1 = numpy.zeros((cart, M, M))
        Dov = numpy.zeros_like(D1)
        for n in range(cart):
            D1_n = D1[n].copy()
            Dov_n = numpy.zeros_like(D1_n)
            F_list = []
            e_list = []
            for cycle in range(max_cycle):
                v1 = vresp([D1_n])
                F1_n = h1[n] + 2*v1
                
                diis_r = F1_n @ D0 @ S0 - S0 @ D0 @ F1_n \
                        + F0 @ D1_n @ S0 - S0 @ D1_n @ F0 \
                        + F0 @ D0 @ s1[n] - s1[n] @ D0 @ F0 
                                                                        
                F_list.append(F1_n)
                e_list.append(diis_r)
                
                F1_n = diis(F_list, e_list, max_diis=6)

                Dov_update = numpy.zeros_like(Dov_n)
                for i in range(nocc):
                    nominator = F1_n - eo[i] * s1[n]
                    for j in range(nvir):
                        Xov = numpy.outer(Co[:,i],Cv[:,j]) 
                        denominator = eo[i] - ev[j]                 
                        alpha = numpy.inner(Co[:,i],(( nominator / denominator) @ Cv[:,j]))                           
                        Dov_update += alpha * Xov 
                
                err = numpy.linalg.norm(Dov_update - Dov_n)
            
                if err < conv_tol:
                    break
                
                alpha = 1
                Dov_n = (1 - alpha) * Dov_n + alpha * Dov_update
            
                D1_n = 2 * Doo[n] + Dov_n - Dov_n.T
                
            D1[n] = D1_n
            Dov[n] = Dov_n
            
            #print(f"n: {n}, cycle: {cycle}, err = {err}, v1 = {numpy.linalg.norm(v1)}")
            
    Dvo = - Dov.transpose(0,2,1)
    
    t1 = time.perf_counter()
    total_time = t1 - t0
    print(f"time spent in DM routine: {total_time}")

    return D1, Doo, Dov, Dvo
        
def solve_dm10_linear(S0, Sinv, F0, D0, S1, F1, method='slv'):
    '''
    Different iterative methods developed for solving the linear
    system of equation returning first-order density matrix in AO basis
    '''
    M = S1.shape[1]
    A, B = build_ab(S0, Sinv, F0, D0)
    
    if method != 'slv':
        R = numpy.kron(numpy.eye(M), A) + numpy.kron(B.T, numpy.eye(M))
    
    D1 = numpy.zeros((3,M,M))       
    for n in range(3):
        
        Q_n = build_q(Sinv, F0, D0, S1[n], F1[n])

        if ( method == 'slv'):
        ########### A X + X B = Q #############
       
            D1_n = linalg.solve_sylvester(A, B, Q_n)

            
        elif ( method == 'gmres'):
        ##### ( I x A + B.T x I ) @ x = q #####
        ########      R           @ x = q #####
    
            q = Q_n.reshape(M**2)
            D1_n, info = gmres(R, q) 
            D1_n = D1_n.reshape((M, M))
        
        elif ( method == 'cg'):
        ####### R.T @ R @ x = R.T @ q #########
        #######    lhs  @ x =   rhs   #########
    
            q = Q_n.reshape(M**2)
            lhs = R.T @ R
            rhs = R.T @ q
        
            D1_n, info = cg(lhs, rhs) 
            D1_n = D1_n.reshape((M, M))
            
        elif ( method == 'cho'):
        
            q = Q_n.reshape(M**2)
            lhs = R.T @ R
            rhs = R.T @ q
            
            lhs_sparse = csc_matrix(lhs)
            factor = cholesky(lhs_sparse)
            D1_n = factor(rhs)
            D1_n = D1_n.reshape((M, M))
            
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from: 'slv', 'gmres', 'cg', 'cho'.")
        
        D1[n] = D1_n
    
    return D1


def solve_dm10_sylvester(nmrobj, mo_coeff=None, mo_occ=None,
                         dm0=None, h1=None, s1=None, with_cphf=None):
    '''Build first-order density matrix in AO basis
        based on Sylvester-DMPT approach'''
        
    t0 = time.perf_counter()
    
    from pyscf.scf import hf    
    mol = nmrobj.mol
    #if mo_coeff is None: mo_coeff = nmrobj._scf.mo_coeff
    #if mo_occ is None: mo_occ = nmrobj._scf.mo_occ
    #if dm0 is None: dm0 = nmrobj._scf.init_guess_by_huckel(mol) #nmrobj._scf.make_rdm1(mo_coeff, mo_occ)
    #if h1 is None: h1 = make_h10(mol, dm0, gauge_orig=nmrobj.gauge_orig)
    #if s1 is None: s1 = make_s10(mol, gauge_orig=nmrobj.gauge_orig) # numpy.eye((h1.shape[1])) 
    if with_cphf is None: with_cphf = nmrobj.cphf
        
    F0, S0, D0, N = get_system(1000, 1000)
    Sinv = numpy.linalg.inv(S0)
    #print("before h1")
    h1, s1 = dummy_F1_S1(F0)
    #print("after h1")
    # S0 = mol.intor('int1e_ovlp')
    # Sinv = numpy.linalg.inv(S0) #inv(S0) 
    #F0 = nmrobj._scf.get_fock(dm=dm0)
    # D0 = .5 * dm0
    ######### property check ##############
    #plt.figure(10)
    #plt.matshow(h1[0])
    print("Cond number of S:", numpy.linalg.cond(S0))

    idem = D0 @ S0 @ D0 - D0
    print("this is idempotent if it's True:", numpy.allclose(idem, numpy.zeros(idem.shape)))
    trace = numpy.trace(D0 @ S0)
    #N = mol.nelectron
    print("this is a correct trace if it's True:", numpy.allclose(trace, N/2), "trace:", trace, "number of occ:", N/2)
    if not with_cphf:
        D1 = solve_dm10_linear(S0, Sinv, F0, D0, s1, h1)
        t1 = time.perf_counter()
        total_time = t1 - t0
        print(f"time spent in DM routine: {total_time}")
        return D1

    else:
        
        vresp = nmrobj._scf.gen_response(singlet=True, hermi=2)
        
        A, B = build_ab(S0, Sinv, F0, D0)
        
        max_cycle = 2000
        conv_tol = 1e-8
        D1 = numpy.zeros_like(s1)
        for n in range(s1.shape[0]):
            S1_n = s1[n]
            D1_n = D1[n]
            F_list = []
            e_list = []
            for cycle in range(max_cycle):
                
                v1 = vresp([D1_n])[0]
                
                F1_n = h1[n] + 2*v1
                                
                diis_r = F1_n @ D0 @ S0 - S0 @ D0 @ F1_n \
                        + F0 @ D1_n @ S0 - S0 @ D1_n @ F0 \
                        + F0 @ D0 @ S1_n - S1_n @ D0 @ F0 \
                                                                        
                F_list.append(F1_n)
                e_list.append(diis_r)
                
                F1_n = diis(F_list, e_list)
                    
                Q_n = build_q(Sinv, F0, D0, S1_n, F1_n)  
                
                D1_update_n = linalg.solve_sylvester(A, B, Q_n)
                err = numpy.linalg.norm(D1_update_n - D1_n)
                 
                if err < conv_tol:
                    print(f"SYL DMPT converged at cycle {cycle} for n {n}")
                    break
                 
                alpha = 1
                D1_n = (1 - alpha) * D1_n + alpha * D1_update_n
                
            D1[n] = D1_n
    t1 = time.perf_counter()
    total_time = t1 - t0
    print(f"time spent in DM routine: {total_time}")
            
    return D1

def tc2_dmpt(F0, F1, S0, S1, Sinv, N,
                      emax, emin,
                      max_cycle=2000, tol=1e-8
                      ):

    I = numpy.eye(S0.shape[0])
    D0 = ((emax * I - Sinv @ F0) @ Sinv) / (emax - emin)
    D1 = (
        - emax * Sinv @ S1 @ Sinv
        - Sinv @ (F1 - S1 @ Sinv @ F0 - F0 @ Sinv @ S1) @ Sinv
        ) / (emax - emin)


    for cycle in range(max_cycle):
        
        ds = D0 @ S0

        delta_N = N - numpy.trace(ds)

        theta = 1 if delta_N > 0 else 0
        alpha = 2 * (theta - 0.5)

        D0_update = D0 + alpha * (D0 - ds @ D0)

        D1_update = D1 + alpha * (
            D1 - (D1 @ ds.T
                  + D0 @ S1 @ D0
                  + ds @ D1)
        )

        err0 = numpy.linalg.norm(D0 - D0_update)
        err1 = numpy.linalg.norm(D1 - D1_update)

        if err0 < tol and err1 < tol:
            #print(f"TC2 DMPT converged at cycle {cycle}")
            break
        
        D0 = D0_update
        D1 = D1_update

    return D0, D1 

def hpcp_dmpt(F0, F1, S0, S1, Sinv, N, 
              emax, emin, 
              max_cycle = 2000, tol=1e-8):
    
    M = F0.shape[0]
    theta = N/M
    mu = numpy.trace(Sinv @ F0) / M     
    beta1 = theta
    beta2 = min( theta/(emax-mu), (1-theta)/(mu-emin) )
    sss = Sinv @ S1 @ Sinv
    sfs = Sinv @ F0 @ Sinv
      
    D0 = beta1 * Sinv + beta2 * ( mu * Sinv - sfs )
    D1 = - beta1 * sss + beta2 * (- mu * sss - Sinv @ F1 @ Sinv + sss @ F0 @ Sinv + Sinv @ F0 @ sss)

    for cycle in range(max_cycle): 
           
        ds = D0 @ S0
        dsd = ds @ D0
        dsdsd = dsd @ S0 @ D0
        c = numpy.trace( (dsd  - dsdsd) @ S0 ) / numpy.trace(ds - dsd @ S0)
        
        D0_update = (1 - 2*c) * D0 + 2*(1 + c) * dsd - 2*dsdsd
        
        D1update = (1 - 2*c) * D1 \
            + 2*(1 + c) * (D1 @ ds.T \
                        + D0 @ S1 @ D0 \
                        + ds @ D1) \
                - 2 * (D1 @ S0 @ dsd \
                       + D0 @ S1 @ dsd \
                           + ds @ D1 @ ds.T \
                        + dsd @ S1 @ D0 \
                        + dsd @ S0 @ D1)
                                
        err0 = numpy.linalg.norm(D0 - D0_update)
                
        err1 = numpy.linalg.norm(D1 - D1update)
        
        if err0 < tol and err1 < tol:
            #print(f"CPHF-HPCP converged at inner cycle {cycle}: ")
            break
                
        D0 = D0_update
        D1 = D1update
        
    return D0, D1

def purification_first_order(nmrobj, method='tc2', max_cycle=2000, tol=1e-8, with_cphf=None):
    
    #from pyscf.scf import hf
    if with_cphf is None: with_cphf = nmrobj.cphf
    #mol = nmrobj.mol
    #F0 = nmrobj._scf.get_fock()
    #mo_occ = nmrobj._scf.mo_occ
    #N = mol.nelectron #numpy.sum(mo_occ > 0)
    #dm0 = nmrobj._scf.init_guess_by_huckel(mol) #dm0 = nmrobj._scf.make_rdm1()
    #F1 = make_h10(mol, dm0, gauge_orig=nmrobj.gauge_orig)
    #S1 = make_s10(mol, gauge_orig=nmrobj.gauge_orig) # numpy.eye((F1.shape[1])) 
    #h1e = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    #vhf = hf.get_veff(mol, dm0)
    #F0 = h1e + vhf
    #S0 =  mol.intor("int1e_ovlp") # numpy.eye((S1.shape[1]))
    F0, S0, D0, N = get_system(1000, 1000)
    #S0 = overlap(10)
    F1, S1 = dummy_F1_S1(F0)
    Sinv = numpy.linalg.inv(S0)
    emax = gershgorin_max(Sinv @ F0)
    emin = gershgorin_min(Sinv @ F0)
        
    D1 = numpy.zeros_like(S1)
    #D0 = .5 * dm0
    print("Cond number of S:", numpy.linalg.cond(S0))
    
    if ( method == 'tc2' ):

        if with_cphf:
            vresp = nmrobj._scf.gen_response(singlet=True, hermi=2)
            
            npert = S1.shape[0]
            
            F_list = []
            R_list = []

            for cphf_cycle in range(max_cycle):

                v_list = vresp([D1[n] for n in range(npert)])

                F1_v = numpy.empty_like(F1)
                R = numpy.empty_like(F1)
                
                ds = D0 @ S0
                df = D0 @ F0
                
                for n in range(npert):
                    F1_v[n] = F1[n] + 2.0 * v_list[n]

                    R[n] = (F1_v[n] @ ds - ds.T @ F1_v[n]
                            + F0 @ D1[n] @ S0    - S0 @ D1[n] @ F0        
                            + df.T @ S1[n]   - S1[n] @ df)

                # if (not numpy.isfinite(R).all()) or (not numpy.isfinite(F1_v).all()):
                #     print(f"CPHF-TC2: non finite F or R at outer cycle {cphf_cycle}")
                #     break

                F_list.append(F1_v)
                R_list.append(R)

                F1_v_new = diis(F_list, R_list)

                D0_new, D1_new = tc2_dmpt(
                    F0, F1_v_new, S0, S1, Sinv, N,
                    emax, emin,
                    max_cycle=max_cycle, tol=tol
                    )

                # if (not numpy.isfinite(D0_new).all()) or (not numpy.isfinite(D1_new).all()):
                #     print(f"CPHF-TC2: non finite D0/D1 at outer cycle {cphf_cycle}")
                #     break

                err1 = numpy.linalg.norm(D0_new - D0)
                err2 = numpy.linalg.norm(D1_new - D1)  

                # print(f"CPHF-TC2 outer cycle {cphf_cycle}: err = {err_cphf:.6e}")

                if err1 < tol and err2 < tol:
                    print(f"CPHF-TC2 converged at outer cycle {cphf_cycle}")
                    break
                
                D0 = D0_new
                D1 = D1_new

        else:
            D0, D1 = tc2_dmpt(
                F0, F1, S0, S1, Sinv, N,
                emax, emin,
                max_cycle=max_cycle, tol=tol
                )

    elif ( method == 'hpcp' ):
        
        if with_cphf: 
            
            vresp = nmrobj._scf.gen_response(singlet=True, hermi=2)
            
            npert = S1.shape[0]
            
            F_list = []
            R_list = []
            for cphf_cycle in range(max_cycle): 
                
                v_list = vresp([D1[n] for n in range(npert)])
                
                F1_v = numpy.empty_like(F1)
                R = numpy.empty_like(F1)
                    
                ds = D0 @ S0
                df = D0 @ F0
                
                for n in range(npert):
                    F1_v[n] = F1[n] + 2.0 * v_list[n]
                    
                    R[n] = (F1_v[n] @ ds - ds.T @ F1_v[n]
                            + F0 @ D1[n] @ S0    - S0 @ D1[n] @ F0        
                            + df.T @ S1[n]   - S1[n] @ df)
                    
                    # if (not numpy.isfinite(R).all()) or (not numpy.isfinite(F1_v).all()):
                    #     print(f"CPHF-HPCP: non finite F or R at outer cycle {cphf_cycle}")
                    #     break
                        
                F_list.append(F1_v.copy())
                R_list.append(R.copy())
                                    
                F1_v_new = diis(F_list, R_list)
                                    
                D0_new, D1_new = hpcp_dmpt(
                        F0, F1_v_new, S0, S1, Sinv, N,
                        emax, emin,
                        max_cycle=max_cycle, tol=tol
                        )
            
                # if (not numpy.isfinite(D0_new).all()) or (not numpy.isfinite(D1_new).all()):
                #     print(f"CPHF-HPCP: non finite D0/D1 at outer cycle {cphf_cycle}")
                #     break
                
                err1 = numpy.linalg.norm(D0_new - D0)
                err2 = numpy.linalg.norm(D1_new - D1)  
                
                #print(f"CPHF-HPCP outer cycle {cphf_cycle}: err1 = {err1:.6e}, err2 = {err2:.6e}")
                
                if err1 < tol and err2 < tol:
                    print(f"CPHF-HPCP converged at outer cycle {cphf_cycle}")
                    break
                    
                D0 = D0_new
                D1 = D1_new
        else:
            D0, D1 = hpcp_dmpt(
                F0, F1, S0, S1, Sinv, N,
                emax, emin,
                max_cycle=max_cycle, tol=tol
                )
                
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: 'tc2', 'hpcp'.")
           
    return D0, D1

# def purification_first_order(nmrobj, method='tc2', max_cycle=50, tol=1e-9, with_cphf=None):

#     mol = nmrobj.mol
#     S0 = mol.intor("int1e_ovlp")
#     F0 = nmrobj._scf.get_fock()
#     mo_occ = nmrobj._scf.mo_occ
#     N = numpy.sum(mo_occ > 0)
#     S1 = make_s10(nmrobj.mol, gauge_orig=nmrobj.gauge_orig)
#     dm0 = nmrobj._scf.make_rdm1()
#     F1 = make_h10(nmrobj.mol, dm0, gauge_orig=nmrobj.gauge_orig)
#     I = numpy.eye(S0.shape[0], dtype=S0.dtype)
#     Sinv = numpy.linalg.inv(S0)
#     emax = gershgorin_max(Sinv @ F0)
#     emin = gershgorin_min(Sinv @ F0)
    
#     if with_cphf:
#         vresp = nmrobj._scf.gen_response(singlet=True, hermi=2)
    
#     if ( method == 'tc2' ):

        
#         D0 = ((emax * I - Sinv @ F0) @ Sinv) / (emax - emin)
#         D1 = (- emax * Sinv @ S1 @ Sinv - Sinv @ ( F1 - S1 @ Sinv @ F0 - F0 @ Sinv @ S1 ) @ Sinv )  / (emax - emin)
#         A = []
#         B = []
#         for cycle in range(max_cycle):

#             trace_DS = numpy.trace(D0 @ S0)
#             delta_N = N - trace_DS
            
#             if delta_N > 0:
#                 theta = 1
#             else:
#                 theta = 0
            
#             alpha = 2 * (theta - 0.5)
                
#             D0_update = D0 + alpha * (D0 - D0 @ S0 @ D0)
                
#             D1update = D1 + alpha * ( D1 - (D1 @ S0 @ D0 \
#                                                       + D0 @ S1 @ D0 \
#                                                           + D0 @ S0 @ D1 ) )
                    
#             err0 = numpy.linalg.norm(D0 - D0_update)
                
#             err1 = numpy.linalg.norm(D1 - D1update)
        
#             if err0 < tol and err1 < tol:
#                 print(f"Converged at cycle {cycle}: ")
#                 break
                
#             D0 = D0_update
#             D1 = D1update
            
#             a = numpy.linalg.norm(D1.transpose(0,2,1)@ S0 + S0 @ D1)
#             print(a)
#             A.append(a)   #A ANTISYMMETRY
                
#             b = numpy.sum(numpy.trace(D1 @ S0, axis1=1, axis2=2) + numpy.trace(D0 @ S1, axis1=1, axis2=2))
#             B.append(b)   #B TRACE  (added S in my trace for correctness and visuality)
            
#     elif ( method == 'hpcp' ):
        
#         M = F0.shape[0]
#         theta = N/M
#         mu = numpy.trace(Sinv @ F0) / M     
#         beta1 = theta
#         beta2 = min( theta/(emax-mu), (1-theta)/(mu-emin) )
#         sss = Sinv @ S1 @ Sinv
#         sfs = Sinv @ F0 @ Sinv
       
#         D0 = beta1 * Sinv + beta2 * ( mu * Sinv - sfs )
#         D1 = - beta1 * sss + beta2 * (- mu * sss - Sinv @ F1 @ Sinv + sss @ F0 @ Sinv + Sinv @ F0 @ sss)
#         A = []
#         B = []
#         for cycle in range(max_cycle): 
           
#             ds = D0 @ S0
#             dsd = ds @ D0
#             dsdsd = dsd @ S0 @ D0
#             c = numpy.trace( (dsd  - dsdsd) @ S0 ) / numpy.trace(ds - dsd @ S0)
               
#             D0_update = (1 - 2*c) * D0 + 2*(1 + c) * dsd - 2*dsdsd
              
#             D1update = (1 - 2*c) * D1 \
#                + 2*(1 + c) * (D1 @ S0 @ D0 \
#                             + D0 @ S1 @ D0 \
#                             + D0 @ S0 @ D1) \
#                        - 2 * (D1 @ S0 @ dsd \
#                             + D0 @ S1 @ dsd \
#                             + ds @ D1 @ S0 @ D0 \
#                             + dsd @ S1 @ D0 \
#                             + dsd @ S0 @ D1)
                                
#             err0 = numpy.linalg.norm(D0 - D0_update)
                
#             err1 = numpy.linalg.norm(D1 - D1update)
        
#             if err0 < tol and err1 < tol:
#                 print(f"Converged at cycle {cycle}: ")
#                 break
                
#             D0 = D0_update
#             D1 = D1update
            
#             a = numpy.linalg.norm(D1.transpose(0,2,1) + D1)
#             A.append(a)
            
#             b = numpy.sum(numpy.trace(D1 @ S0, axis1=1, axis2=2) + numpy.trace(D0 @ S1, axis1=1, axis2=2))
#             B.append(b)
            
#     else:
#         raise ValueError(f"Unknown method '{method}'. Choose from: 'tc2', 'hpcp'.")
           
#     return D0, D1, A, B

def get_fock(nmrobj, dm0=None, gauge_orig=None):
    r'''First order partial derivatives of Fock matrix wrt external magnetic
    field.  \frac{\partial F}{\partial B}
    '''
    if dm0 is None: dm0 = nmrobj._scf.make_rdm1()
    if gauge_orig is None: gauge_orig = nmrobj.gauge_orig

    log = logger.Logger(nmrobj.stdout, nmrobj.verbose)
    h1 = make_h10(nmrobj.mol, dm0, gauge_orig, log)
    if nmrobj.chkfile:
        lib.chkfile.dump(nmrobj.chkfile, 'nmr/h1', h1)
    return h1

def gen_vind(mf, mo_coeff, mo_occ):
    '''Induced potential'''
    vresp = mf.gen_response(singlet=True, hermi=2)

    occidx = mo_occ > 0
    orbo = mo_coeff[:,occidx]
    nocc = orbo.shape[1]
    nao, nmo = mo_coeff.shape
    def vind(mo1):
        dm1 = [reduce(numpy.dot, (mo_coeff, x*2, orbo.T.conj()))
               for x in mo1.reshape(-1,nmo,nocc)]
        dm1 = numpy.asarray([d1-d1.conj().T for d1 in dm1])
        v1mo = lib.einsum('xpq,pi,qj->xij', vresp(dm1), mo_coeff.conj(), orbo)
        return v1mo.ravel()
    return vind


class NMR(lib.StreamObject):
    def __init__(self, scf_method):
        self.mol = scf_method.mol
        self.verbose = scf_method.mol.verbose
        self.stdout = scf_method.mol.stdout
        self.chkfile = scf_method.chkfile
        self._scf = scf_method

        self.shielding_nuc = range(self.mol.natm)
# gauge_orig=None will call GIAO. A coordinate array leads to common gauge
        self.gauge_orig = None
        self.cphf = True
        self.max_cycle_cphf = 20
        self.conv_tol = 1e-9

        self.mo10 = None
        self.mo_e10 = None
        self.dm10 = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s for %s ********',
                 self.__class__, self._scf.__class__)
        if self.gauge_orig is None:
            log.info('gauge = GIAO')
        else:
            log.info('Common gauge = %s', str(self.gauge_orig))
        log.info('shielding for atoms %s', str(self.shielding_nuc))
        if self.cphf:
            log.info('Solving MO10 eq with CPHF.')
            log.info('CPHF conv_tol = %g', self.conv_tol)
            log.info('CPHF max_cycle_cphf = %d', self.max_cycle_cphf)
        if not self._scf.converged:
            log.warn('Ground state SCF is not converged')
        return self

    # Note mo10 is the imaginary part of MO^1
    def kernel(self, mo1=None):
        return self.shielding(mo1)
    def shielding(self, mo1=None, dm1=None, method='mo1'):
        cput0 = (logger.process_clock(), logger.perf_counter())
        _, t0 = cput0
        self.check_sanity()
        self.dump_flags()

        unit_ppm = nist.ALPHA**2 * 1e6
        msc_dia = self.dia(self.gauge_orig)
        
        if ( method == 'mcw'):
            if dm1 is None:
                self.dm10 = self.solve_dm10_mcweeny()[0]
                dm1 = self.dm10
            msc_para = self.para_dm(dm10=dm1)
        
        elif ( method == 'slv'):
            if dm1 is None:
                self.dm10 = self.solve_dm10_sylvester()
                dm1 = self.dm10
            msc_para = self.para_dm(method = 'slv', dm10=dm1)
        elif ( method == 'tc2'):
            if dm1 is None:
                self.dm10 = self.purification_first_order()[1]
                dm1 = self.dm10
            msc_para = self.para_dm(method = 'tc2', dm10=dm1)
        elif ( method == 'hpcp'):
            if dm1 is None:
                self.dm10 = self.purification_first_order(method='hpcp')[1]
                dm1 = self.dm10
            msc_para = self.para_dm(method = 'hpcp', dm10=dm1)
        else:
            if mo1 is None:
                self.mo10, self.mo_e10 = self.solve_mo1()
                mo1 = self.mo10
            msc_para, para_vir, para_occ = self.para(mo10=mo1)

        msc_dia *= unit_ppm
        msc_para *= unit_ppm
        #para_vir *= unit_ppm
        #para_occ *= unit_ppm
        e11 = msc_para + msc_dia

        logger.timer(self, 'NMR shielding', *cput0)
        if self.verbose >= logger.NOTE:
            for i, atm_id in enumerate(self.shielding_nuc):
                _write(self.stdout, e11[i],
                       '\ntotal shielding of atom %d %s'
                       % (atm_id, self.mol.atom_symbol(atm_id)))
                _write(self.stdout, msc_dia[i], 'dia-magnetic contribution')
                _write(self.stdout, msc_para[i], 'para-magnetic contribution')
#                if self.verbose >= logger.INFO:
#                    _write(self.stdout, para_occ[i], 'occ part of para-magnetism')
#                    _write(self.stdout, para_vir[i], 'vir part of para-magnetism')
        t1 = logger.perf_counter()
        total_time = t1 - t0
        print(f"\ntime spent in shielding in seconds: {total_time}")

        return e11

    dia = dia
    para = para
    para_dm = para_dm
    get_fock = get_fock
    solve_mo1 = solve_mo1
    solve_dm10_mcweeny = solve_dm10_mcweeny
    solve_dm10_sylvester = solve_dm10_sylvester
    purification_first_order = purification_first_order

    def get_ovlp(self, mol=None, gauge_orig=None):
        if mol is None: mol = self.mol
        if gauge_orig is None: gauge_orig = self.gauge_orig
        return get_ovlp(mol, gauge_orig)

from pyscf import scf
scf.hf.RHF.NMR = lib.class_as_method(NMR)

def _write(stdout, msc3x3, title):
    stdout.write('%s\n' % title)
    stdout.write('B_x %s\n' % str(msc3x3[0]))
    stdout.write('B_y %s\n' % str(msc3x3[1]))
    stdout.write('B_z %s\n' % str(msc3x3[2]))
    stdout.flush()


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 1
    mol.output = None

    mol.atom.extend([
        [1   , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ])
    mol.nucmod = {'F': 2} # gaussian nuclear model
    mol.basis = {'H': '6-31g',
                 'F': '6-31g',}
    mol.build()

    rhf = scf.RHF(mol).run()
    nmr = rhf.NMR()
    nmr.cphf = True
    #nmr.gauge_orig = (0,0,0)
    msc = nmr.kernel() # _xx,_yy = 375.232839, _zz = 483.002139
    print(msc[1][0,0], msc[1][1,1], 375.232839)
    print(msc[1][2,2], 483.002139)
    print(lib.finger(msc) - -132.22895063293751)

    nmr.cphf = True
    nmr.gauge_orig = (1,1,1)
    msc = nmr.shielding()
    print(msc[1][0,0], msc[1][1,1], 342.447242)
    print(msc[1][2,2], 483.002139)
    print(lib.finger(msc) - -108.48528212325664)

    nmr.cphf = False
    nmr.gauge_orig = None
    msc = nmr.shielding()
    print(msc[1][0,0], msc[1][1,1], 449.032227)
    print(msc[1][2,2], 483.002139)
    print(lib.finger(msc) - -133.26526049655627)

    mol.atom.extend([
        [1 , (1. , 0.3, .417)],
        [1 , (0.2, 1. , 0.)],])
    mol.build()
    mf = scf.RHF(mol).run()
    nmr = NMR(mf)
    nmr.cphf = False
    nmr.gauge_orig = None
    msc = nmr.shielding()
    print(msc[1][0,0], 283.514599)
    print(msc[1][1,1], 292.578151)
    print(msc[1][2,2], 257.348176)
    print(lib.finger(msc) - -123.98600632099961)
    


from pyscf import gto, scf
from pyscf.prop.nmr.rhf_dm import NMR
import matplotlib.pyplot as plt
import numpy as np
from pyscf.data import nist
nist.ALPHA = 1./137.03599967994

mol = gto.Mole()
mol.verbose = 1
# mol.atom = [
#     ["C", (-2.10, 0.00, 0.00)],
#     ["H", (-2.70, 0.92, 0.00)],
#     ["H", (-2.70, -0.92, 0.00)],
#     ["C", (-0.70, 0.00, 0.00)],
#     ["H", (-0.10, 0.92, 0.00)],
#     ["C", (0.70, 0.00, 0.00)],
#     ["H", (1.30, -0.92, 0.00)],
#     ["C", (2.10, 0.00, 0.00)],
#     ["H", (2.70, 0.92, 0.00)],
#     ["H", (2.70, -0.92, 0.00)],
# ]
# mol.basis = {
#     "C": '6-31g',
#     "H": '6-31g',
# }
# mol.build()

mol.atom = [
    [1   , (0. , 0. , .917)],
    ["F" , (0. , 0. , 0.)], ]
mol.basis = {"H": 'cc_pvdz',
             "F": 'cc_pvdz',}
mol.build()


mf = scf.RHF(mol)
# mf.conv_tol_grad = 1e-10  ############ KEEP VERY HIGH !! IMPORTANT !! ######################
# mf.conv_tol = 1e-14
# mf.scf()

nmr = NMR(mf)
nmr.cphf = False
nmr.gauge_orig = None

D1_syl = solve_dm10_sylvester(nmr)
D1_mcw, _, _, _ = solve_dm10_mcweeny(nmr)

_, D1_tc2 = purification_first_order(nmr, method='tc2')
_, D1_hpcp = purification_first_order(nmr, method='hpcp')

print('Norm of the density matrix D1_hpcp:', np.linalg.norm(D1_hpcp))
print('Norm of the density matrix D1_tc2:', np.linalg.norm(D1_tc2))


test_D1 = D1_tc2 - D1_hpcp
print('Comparing D1_tc2 with D1_hpcp:', np.linalg.norm(test_D1.reshape(-1, test_D1.shape[-1]), ord='fro'))
test_D1 = D1_hpcp - D1_syl
print('Comparing D1_hpcp with D1_syl:', np.linalg.norm(test_D1.reshape(-1, test_D1.shape[-1]), ord='fro'))
test_D1 = D1_syl - D1_mcw
print('Comparing D1_syl with D1_mcw:', np.linalg.norm(test_D1.reshape(-1, test_D1.shape[-1]), ord='fro'))

# import matplotlib.pyplot as plt 

# plt.figure(0)
# plt.matshow(D1_mcw[0])

# plt.figure(1)
# plt.matshow(D1_syl[0])

# plt.figure(2)
# plt.matshow(D1_hpcp[0])

# msc_para = para_dm(nmr, method='tc2')
# msc_para_m10, _, _ = para(nmr)
# frobenius_diff = np.linalg.norm(msc_para - msc_para_m10)
# print("Frobenius difference:", frobenius_diff)

# s0 = mol.intor("int1e_ovlp")
# A_tc2 = np.array(A_tc2)
# B_tc2 = np.array(B_tc2)
# A_hpcp = np.array(A_hpcp)
# B_hpcp = np.array(B_hpcp)

# iterations_TC2 = np.arange(len(A_tc2))
# iterations_HPCP = np.arange(len(A_hpcp))


# plt.figure(figsize=(6.4, 4.8) )
# plt.plot(iterations_TC2, A_tc2, marker='o', label='TC2')
# plt.plot(iterations_HPCP, A_hpcp, marker='s', label='HPCP')
# plt.xlabel("Iteration")
# plt.ylabel("Trace Deviation")
# plt.legend()
# plt.tight_layout()
# plt.savefig("trace_comparison.png")

# plt.figure(figsize=(6.4, 4.8) )
# plt.plot(iterations_TC2, B_tc2, marker='o', label='TC2')
# plt.plot(iterations_HPCP, B_hpcp, marker='s', label='HPCP')
# plt.xlabel("Iteration")
# plt.ylabel("Antisymmetry Norm")
# plt.legend()
# plt.tight_layout()
# plt.savefig("antisymmetry_comparison.png")
