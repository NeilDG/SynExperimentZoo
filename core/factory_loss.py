import torch.nn as nn
from losses import common_losses, vgg_loss, bicubic_loss
import kornia.losses

class LossFactory:
    @staticmethod
    def get_losses(config, device):
        """
        Returns a LossRepository with weights and behaviors defined by experiment config.
        """
        loss_repo = common_losses.LossRepository(device)
        
        # Inject weights into loss repo from config
        # Expecting config to have experiment.losses dictionary
        weights = config.get('experiment.losses', {})
        
        return loss_repo, weights
