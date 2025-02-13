#!/usr/bin/env python

'''
Computing NMR shielding constants
'''
import os

os.environ["PYSCF_EXT_PATH"] = "/Users/lioneltruflandier/pyscf-on/properties-on"

from pyscf import gto, dft, scf
from pyscf.prop import nmr
from pyscf.data import nist
import numpy as np

mol = gto.M(atom='''
            C 0 0 0
            O 0 0 1.1747
            ''',
            basis='sto-3g', verbose=20)

print('### M (basis set size), Ne (nb. of electrons), N (nb. of occupied states)')
print(mol.nao_nr(),mol.tot_electrons(),mol.tot_electrons()/2)

conv, e, mo_e, mo, mo_occ = scf.hf.kernel(scf.hf.SCF(mol))

occidx = mo_occ > 0
viridx = mo_occ == 0


print('### occidx, viridx')
print(occidx)
print(viridx)

e_a = mo_e[viridx]
print('### e_a: 1D e_a(i) size (M-N)', np.shape(e_a))
print(e_a)

e_i = mo_e[occidx]
print('### e_i: 1D e_i(i) size N', np.shape(e_i))
print(e_i)

e_ai = (e_a.reshape(-1,1) - e_i)
print('### e_ai: 2D 1/(e_a(i) - (e_i(j))) size (M-N)xN',np.shape(e_ai))
print(e_ai)

e_ji = e_i.reshape(-1,1) - e_i
print('### e_ji: 2D 1/(e_j(i) - (e_i(j))) size NxN',np.shape(e_ji))
print(e_ji)

#nmr.RHF(mf).kernel()
mf = scf.RHF(mol)
print('### ', mf)
mf.kernel()

#mo10, mo_e10 = nmr.RHF_DM(mf).solve_mo1(with_cphf=False)

#mo10, mo_e10 = nmr.rhf_dm.solve_mo1(mf,with_cphf=False)
#nmr.rhf_dm.make_s10(mol)

#m.kernel()


msc_dia = nmr.RHF_DM(mf).dia()
msc_para, para_vir, para_occ = nmr.RHF(mf).para()

total = msc_dia + msc_para
print(total*nist.ALPHA**2 * 1e6)