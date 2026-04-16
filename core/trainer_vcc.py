import torch
import torch.amp as amp
from core.factory_model import ModelFactory
from core.factory_loss import LossFactory

class ScalableTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Triple-Factory Initialization
        self.netG, self.netD = ModelFactory.create_model(config, device)
        self.loss_repo, self.loss_weights = LossFactory.get_losses(config, device)
        
        # Auto-Gradient Accumulation logic
        load_size = config.get('training.load_size', 1)
        batch_size = config.get('training.batch_size', 1)
        self.accum_steps = max(1, batch_size // load_size)
        print(f"Auto-Gradient Accumulation: {self.accum_steps} steps (Load: {load_size}, Batch: {batch_size})")

        # Optimizer setup
        g_lr = config.get('experiment.hyperparams.g_lr', 0.0002)
        d_lr = config.get('experiment.hyperparams.d_lr', 0.0002)
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=g_lr)
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=d_lr)
        
        self.scaler = amp.GradScaler()
        self.iteration = 0

    def train_step(self, input_map):
        img_a = input_map["img_a"]
        img_b = input_map["img_b"]
        
        with amp.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu"):
            # Generator Forward
            output = self.netG(img_a)
            
            # Compute Losses (Simplified for demonstration)
            l1_weight = self.loss_weights.get('l1_weight', 1.0)
            l1_loss = self.loss_repo.l1_loss(output, img_b) * l1_weight
            
            # Gradient Accumulation Scaling
            total_loss = l1_loss / self.accum_steps
            
        self.scaler.scale(total_loss).backward()
        
        self.iteration += 1
        
        # Step optimizers after accum_steps
        if self.iteration % self.accum_steps == 0:
            self.scaler.step(self.optimizerG)
            self.scaler.update()
            self.optimizerG.zero_grad()
            
        return total_loss.item()
