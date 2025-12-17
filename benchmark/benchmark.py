import sys
import pyscf
from pyscf import gto, scf
from pyscf import lib
import numpy
import importlib.util
print(importlib.util.find_spec('pyscf'))
from pyscf.prop.nmr.rhf_dm import NMR

import psutil
import time

def log_usage(tag=""):
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 ** 2)  # MB
    print(f"[{tag}] Memory usage: {mem:.2f} MB")

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

    start = time.time()
    log_usage("Before MO")
    msc = nmr.shielding()
    log_usage("After MO")
    print(f"[MO] Time: {time.time() - start:.2f} s")

    start = time.time()
    log_usage("Before MCW")
    msc_dm = nmr.shielding(method='mcw')
    log_usage("After MCW")
    print(f"[MCW] Time: {time.time() - start:.2f} s")

    start = time.time()
    log_usage("Before SLV")
    msc_s = nmr.shielding(method='slv')
    log_usage("After SLV")
    print(f"[SLV] Time: {time.time() - start:.2f} s")

    start = time.time()
    log_usage("Before TC2")
    msc_tc2 = nmr.shielding(method='tc2')
    log_usage("After TC2")
    print(f"[TC2] Time: {time.time() - start:.2f} s")

    start = time.time()
    log_usage("Before HPCP")
    msc_hpcp = nmr.shielding(method='hpcp')
    log_usage("After HPCP")
    print(f"[HPCP] Time: {time.time() - start:.2f} s")


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
