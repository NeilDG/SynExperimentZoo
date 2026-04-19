import unittest
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config_parser import ConfigParser

class TestSimplifiedVCC(unittest.TestCase):
    def test_mobisr_loading(self):
        vcc = "mobisr_v02.06_div2k.12.3"
        cp = ConfigParser(vcc)
        
        # Test parsing
        self.assertEqual(cp.problem, "mobisr")
        self.assertEqual(cp.version, "v02.06_div2k")
        self.assertEqual(cp.hyper_id, "12")
        self.assertEqual(cp.loss_id, "3")
        
        # Load config for RTX 3090 (index 3)
        config = cp.load_config(server_config_index=3)
        
        # Test common param
        self.assertEqual(config['input_nc'], 3)
        
        # Test version param
        self.assertEqual(config['model_type'], 4)
        self.assertEqual(config['num_blocks'], 3)
        
        # Test server path injection and resolution
        self.assertEqual(config['server_path'], "X:/SuperRes Dataset/")
        self.assertEqual(config['low_path'], "X:/SuperRes Dataset/div2k/lr/*.png")
        
        # Test hyperparams
        self.assertEqual(config['hyperparams']['g_lr'], 0.0004)
        
        # Test losses
        self.assertEqual(config['losses']['l1_weight'], 0.0)
        self.assertEqual(config['losses']['perceptual_weight'], 1.0)

    def test_v_prefix_compatibility(self):
        vcc = "V.mobisr_v02.05_div2k.10.1"
        cp = ConfigParser(vcc)
        self.assertEqual(cp.problem, "mobisr")
        config = cp.load_config(0)
        self.assertEqual(config['model_type'], 1)

if __name__ == '__main__':
    unittest.main()
