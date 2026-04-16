import os
import sys
import traceback
import types

import torch

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# If CUDA is not available, make amp.autocast a no-op to avoid device errors
if not torch.cuda.is_available():
    class _NoOpCast:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
    import torch.amp
    torch.amp.autocast = _NoOpCast

# Patch VGG perceptual loss to avoid internet downloads during unit tests
from losses import vgg_loss as _vgg_loss

class _DummyVGG:
    def __init__(self, *args, **kwargs):
        pass
    def to(self, device):
        return self
    def eval(self):
        return self
    def __call__(self, a, b):
        # return a zero scalar tensor on the same device
        return torch.zeros((), device=a.device, dtype=a.dtype)

_vgg_loss.VGGPerceptualLoss = _DummyVGG

# Now import project modules that depend on losses
import global_config
from config.network_config import ConfigHolder
from trainers import paired_trainer


def make_minimal_configs(model_type: int):
    # Minimal network config required by the pipeline
    net_cfg = {
        'model_type': model_type,
        'input_nc': 3,
        'num_blocks': 3,
        'batch_size': [2, 2, 2, 2],
        'load_size': [2, 2, 2, 2],
        'min_epochs': 1,
        'max_epochs': 1,
        # dataset values are not used in this smoke test path, but some code expects them
        'dataset_version': 'div2k',
        'low_path': '/lr/*.png',
        'high_path': '/bicubic_x4/*.png',
    }

    # Single set of hyperparameters (index 0)
    hyper_cfg = {
        'hyperparams': [
            {
                'g_lr': 1e-4,
                'd_lr': 1e-4,
                'dropout_rate': 0.0,
                'norm_mode': 'batch',
            }
        ]
    }

    # Single set of loss weights (index 0). Keep only L1 active to exercise backprop.
    loss_cfg = {
        'loss_weights': [
            {
                'l1_weight': 1.0,
                'perceptual_weight': 0.0,
                'color_weight': 0.0,
                'tv_weight': 0.0,
                'bicubic_weight': 0.0,
                'adv_weight': 1.0,
                'is_bce': 0,
            }
        ]
    }

    return net_cfg, hyper_cfg, loss_cfg


def run_single_smoke(model_type: int, device: torch.device) -> bool:
    # Reset shared config holder
    ConfigHolder.destroy()

    net_cfg, hyper_cfg, loss_cfg = make_minimal_configs(model_type)
    ConfigHolder.initialize(net_cfg, hyper_cfg, loss_cfg)

    # Configure global runtime knobs expected by the trainer
    global_config.sr_network_version = f"unittest_modeltype_{model_type}"
    global_config.hyper_iteration = 0
    global_config.loss_iteration = 0
    global_config.load_size = 2
    global_config.batch_size = 2
    global_config.plot_enabled = 0  # disable visdom

    try:
        trainer = paired_trainer.PairedTrainer(device)

        # Create a tiny random batch (simulate low/high pairs)
        a = torch.rand((global_config.load_size, 3, 64, 64), device=device)
        b = torch.rand((global_config.load_size, 3, 64, 64), device=device)
        input_map = {"img_a": a, "img_b": b}

        # One training step (epoch=0, iteration=1)
        trainer.train(0, 1, input_map)
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        return True
    except Exception:
        traceback.print_exc()
        return False


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Candidate recent models integrated in the SR pipeline
    candidates = {
        4: 'FFA Net',
        5: 'RRDBNet',
        6: 'SwinIR',
    }

    results = {}
    for mt, name in candidates.items():
        print(f"\n[SMOKE] Testing model_type={mt} ({name})...")
        ok = run_single_smoke(mt, device)
        results[name] = ok
        print(f"[RESULT] {name}: {'OK' if ok else 'FAILED'}")

    print("\nSummary:")
    for name, ok in results.items():
        print(f" - {name}: {'OK' if ok else 'FAILED'}")

    # Non-zero exit on failure
    if not all(results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
