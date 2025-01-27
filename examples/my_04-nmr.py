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
            basis='ccpvdz', verbose=20)

mf = scf.RHF(mol)
mf.kernel()

nmr.RHF(mf).kernel()

nmr.RHF_DM(mf).kernel()

#msc_dia = nmr.RHF(mf).dia()
#msc_para, para_vir, para_occ = nmr.RHF(mf).para()

#total = msc_dia + msc_para
#print(total*nist.ALPHA**2 * 1e6)