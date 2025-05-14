import unittest
from pyscf import gto, scf
from pyscf.prop.nmr.rhf_dm import NMR
from pyscf.data import nist
nist.ALPHA = 1./137.03599967994

mol = gto.Mole()
mol.verbose = 20
mol.output = '/dev/null'

mol.atom = [
    [1   , (0. , 0. , .917)],
    ["F" , (0. , 0. , 0.)], ]
mol.basis = {"H": 'cc_pvdz',
             "F": 'cc_pvdz',}
mol.build()

mf = scf.RHF(mol)
mf.conv_tol_grad = 1e-6
mf.conv_tol = 1e-14
mf.scf()

def finger(mat):
    return abs(mat).sum()

class TestNMRShielding(unittest.TestCase):

    def test_giao(self):
        nmr = NMR(mf)
        nmr.cphf = False
        nmr.gauge_orig = None
        msc = nmr.shielding()
        self.assertAlmostEqual(finger(msc), 1488.0948832003237, 4)

    def test_common_gauge(self):
        nmr = NMR(mf)
        nmr.cphf = False
        nmr.gauge_orig = (1,1,1)
        msc = nmr.shielding()
        self.assertAlmostEqual(finger(msc), 1636.7413245248476, 4)

    def test_mcw_method(self):
        nmr = NMR(mf)
        nmr.cphf = False
        nmr.gauge_orig = None
        msc = nmr.shielding(method='mcw')
        self.assertAlmostEqual(finger(msc), 1488.0948832003241, 4)
        
    def test_mcw_method_common_gauge(self):
        nmr = NMR(mf)
        nmr.cphf = False
        nmr.gauge_orig = (1,1,1)
        msc = nmr.shielding(method='mcw')
        self.assertAlmostEqual(finger(msc), 1636.741324524848, 4)

    def test_sylvester_method(self):
        nmr =  NMR(mf)
        nmr.cphf = False
        nmr.gauge_orig = None
        msc = nmr.shielding(method='slv')
        self.assertAlmostEqual(finger(msc), 1488.09488323189, 4)
    
    def test_sylvester_method_common_gauge(self):
        nmr =  NMR(mf)
        nmr.cphf = False
        nmr.gauge_orig = (1,1,1)
        msc = nmr.shielding(method='slv')
        self.assertAlmostEqual(finger(msc), 1636.741324525055, 4)
        
    def test_tc2_method(self):
        nmr =  NMR(mf)
        nmr.cphf = False
        nmr.gauge_orig = None
        msc = nmr.shielding(method='tc2')
        self.assertAlmostEqual(finger(msc), 1488.0948814407607, 4)
    
    def test_tc2_method_common_gauge(self):
        nmr =  NMR(mf)
        nmr.cphf = False
        nmr.gauge_orig = (1,1,1)
        msc = nmr.shielding(method='tc2')
        self.assertAlmostEqual(finger(msc), 1636.741322902088, 4)
        
    def test_hpcp_method(self):
        nmr =  NMR(mf)
        nmr.cphf = False
        nmr.gauge_orig = None
        msc = nmr.shielding(method='hpcp')
        self.assertAlmostEqual(finger(msc), 1488.0948831895964, 4)
    
    def test_hpcp_method_common_gauge(self):
        nmr =  NMR(mf)
        nmr.cphf = False
        nmr.gauge_orig = (1,1,1)
        msc = nmr.shielding(method='hpcp')
        self.assertAlmostEqual(finger(msc), 1636.7413245029861, 4)
        
    def test_giao_cphf(self):
        nmr = NMR(mf)
        nmr.cphf = True
        nmr.gauge_orig = None
        msc = nmr.shielding()
        self.assertAlmostEqual(finger(msc), 1358.9823388784284, 4)

    def test_common_gauge_cphf(self):
        nmr = NMR(mf)
        nmr.cphf = True
        nmr.gauge_orig = (1,1,1)
        msc = nmr.shielding()
        self.assertAlmostEqual(finger(msc), 1562.3864371779719, 4)

    def test_mcw_method_cphf(self):
        nmr = NMR(mf)
        nmr.cphf = True
        nmr.gauge_orig = None
        msc = nmr.shielding(method='mcw')
        self.assertAlmostEqual(finger(msc), 1358.9826194901505, 4)
        
    def test_mcw_method_common_gauge_cphf(self):
        nmr = NMR(mf)
        nmr.cphf = True
        nmr.gauge_orig = (1,1,1)
        msc = nmr.shielding(method='mcw')
        self.assertAlmostEqual(finger(msc), 1562.3859181650841, 4)

    def test_sylvester_method_cphf(self):
        nmr =  NMR(mf)
        nmr.cphf = True
        nmr.gauge_orig = None
        msc = nmr.shielding(method='slv')
        self.assertAlmostEqual(finger(msc), 1358.9826179886654, 4)
    
    def test_sylvester_method_common_gauge_cphf(self):
        nmr =  NMR(mf)
        nmr.cphf = True
        nmr.gauge_orig = (1,1,1)
        msc = nmr.shielding(method='slv')
        self.assertAlmostEqual(finger(msc), 1562.3859191742165, 4)

if __name__ == "__main__":
    print("Running RHF-DM NMR shielding tests...")
    unittest.main()
