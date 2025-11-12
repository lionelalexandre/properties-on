import sys
import os
import pyscf
from pyscf import gto, scf
from pyscf import lib
import numpy
import importlib.util
print(importlib.util.find_spec('pyscf'))
os.environ["PYSCF_EXT_PATH"] = "/Users/lioneltruflandier/pyscf-on/properties-on"
from pyscf.prop.nmr.rhf_dm import NMR



# Directory containing molecule xyz files
molecules_dir = 'xyz'

# List of molecule xyz files
molecules_list = 'benchmark.list'

# Open molecule_list and append molecule files
molecules = [ ]
f = open(molecules_list,'r')
for line in f :
    molecules.append(molecules_dir+'/'+line.strip())

# Basis set and convergence settings
basis_set = '6-31g'
conv_tolerance = 1e-14

# Loop over each molecule
for mol_file in molecules:
    print(f"\nProcessing molecule: {mol_file}")

    mol = gto.Mole()
    mol.atom = open(mol_file).read()  # Read atomic positions
    mol.basis = basis_set
    mol.verbose = 4
    mol.build()

    # diagonalization
    mf = scf.RHF(mol).set(conv_tol=conv_tolerance, conv_check=True)
    mf.kernel(dmp_scf=False)

    nmr = NMR(mf)
    nmr.cphf = False
    nmr.gauge_orig = None

    msc = nmr.shielding()

    msc_dm = nmr.shielding(method='mcw')

    msc_s = nmr.shielding(method='slv')

    msc_tc2 = nmr.shielding(method='tc2')

    msc_hpcp = nmr.shielding(method='hpcp')

    # print results
    test_D1 = msc - msc_dm
    frobenius_norm = numpy.linalg.norm(test_D1.reshape(-1, test_D1.shape[-1]), ord='fro')
    print(f"Frobenius norm difference: {frobenius_norm:.9e}")

    test_D1 = msc - msc_s
    frobenius_norm = numpy.linalg.norm(test_D1.reshape(-1, test_D1.shape[-1]), ord='fro')
    print(f"Frobenius norm difference: {frobenius_norm:.9e}")

    test_D1 = msc - msc_tc2
    frobenius_norm = numpy.linalg.norm(test_D1.reshape(-1, test_D1.shape[-1]), ord='fro')
    print(f"Frobenius norm difference: {frobenius_norm:.9e}")

    test_D1 = msc - msc_hpcp
    frobenius_norm = numpy.linalg.norm(test_D1.reshape(-1, test_D1.shape[-1]), ord='fro')
    print(f"Frobenius norm difference: {frobenius_norm:.9e}")

    print('Norms of msc, msc_dm, msc_s, msc_tc2, msc_hpcp:', numpy.linalg.norm(msc.reshape(-1, msc.shape[-1]), ord='fro'), \
	   numpy.linalg.norm(msc_dm.reshape(-1, msc_dm.shape[-1]), ord='fro'), numpy.linalg.norm(msc_s.reshape(-1, msc_s.shape[-1]), ord='fro'), \
	   numpy.linalg.norm(msc_tc2.reshape(-1, msc_tc2.shape[-1]), ord='fro'), numpy.linalg.norm(msc_hpcp.reshape(-1, msc_hpcp.shape[-1]), ord='fro'))
