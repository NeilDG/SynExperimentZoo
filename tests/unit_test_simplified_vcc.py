import unittest
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config_parser import ConfigParser

class TestSimplifiedVCC(unittest.TestCase):
    def test_mobisr_loading_convention(self):
        # Convention: YY in vXX.YY matches model ID
        vcc = "mobisr_v02.04_div2k.12.3"
        cp = ConfigParser(vcc)
        config = cp.load_config(server_config_index=3)
        
        self.assertEqual(config['model_type'], 4)
        self.assertEqual(config['num_blocks'], 3)
        self.assertEqual(config['server_path'], "X:/SuperRes Dataset/")

    def test_mobisr_new_mapping(self):
        # User requested: v02.06 = model 6
        vcc = "mobisr_v02.06_div2k.12.3"
        cp = ConfigParser(vcc)
        config = cp.load_config(server_config_index=3)
        
        self.assertEqual(config['model_type'], 6)
        self.assertEqual(config['num_blocks'], 6)

    def test_v_prefix_compatibility(self):
        # User requested: v02.01 = model 1
        vcc = "V.mobisr_v02.01_div2k.1.1"
        cp = ConfigParser(vcc)
        config = cp.load_config(0)
        self.assertEqual(config['model_type'], 1)

    def test_fcg2cityscapes_loading(self):
        vcc = "fcg2cityscapes_v00.01.1.1"
        cp = ConfigParser(vcc)
        config = cp.load_config(3)
        self.assertEqual(config['model_type'], 1)
        self.assertEqual(config['dataset_a_train'], "X:/Segmentation Dataset/FCG-Synth-01-patched/train-rgb/sequence.0/*.png")

if __name__ == '__main__':
    unittest.main()
