#import os
#os.environ["PYSCF_EXT_PATH"] = "/home/teodora/internship/pyscf-on/properties-on"

import sys
import pyscf
from pyscf import gto, scf
from pyscf import lib
from pyscf.prop import nmr
import numpy
import importlib.util
print(importlib.util.find_spec('pyscf'))
from rhf_dm import NMR

# Directory containing molecule xyz files
molecules_dir = 'xyz'

# List of molecule xyz files
molecules_list = 'benchmark.list'

# Open molecule_list and append molecule files
molecules = [ ]
f = open(molecules_list,'r')
for line in f :
    molecules.append(molecules_dir+'/'+line.strip())

# List of molecule xyz files
#molecules = ['benzene.xyz', 'naphtalene.xyz', 'anthracene.xyz', 'tetracene.xyz', 'pentacene.xyz', 'hexacene.xyz', 'heptacene.xyz', 'octacene.xyz', 'nonacene.xyz']#,decacene.xyz]

# Basis set and convergence settings
basis_set = '6-31G'
conv_tolerance = 1e-10

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

    print("mo10:")
    # shielding with mo10
    nmr = NMR(mf)
    nmr.cphf = False
    nmr.gauge_orig = None
    msc = nmr.shielding()

    print("dm10:")
    # shielding with dm10
    nmr_2 = NMR(mf)
    nmr_2.cphf = False
    nmr_2.gauge_orig = None
    msc_dm = nmr_2.shielding(use_dm10=True)

    # purification
    #mf = scf.RHF_DM(mol).set(conv_tol=conv_tolerance, conv_check=True)
    #mf.kernel(dmp_scf=True)
    #e_tot_rhf_dm = mf.energy_tot()

    # Print results
    print("\nComparing msc with msc_dm (expected perturbed density matrix)...")
    test_D1 = msc - msc_dm
    frobenius_norm = numpy.linalg.norm(test_D1.reshape(-1, test_D1.shape[-1]), ord='fro')
    print(f"Frobenius norm difference: {frobenius_norm:.6e} (should be close to zero)")
