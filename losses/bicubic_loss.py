import torch
import torch.nn as nn
import torch.nn.functional as F


class BicubicDegradationLoss(nn.Module):
    def __init__(self, scale_factor=4, loss_type='L1'):
        super(BicubicDegradationLoss, self).__init__()
        self.scale_factor = scale_factor
        if loss_type == 'L1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'L2':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError("loss_type must be either 'L1' or 'L2'")

    def forward(self, sr_image, hr_image):
        # Step 1: Bicubic downsample the HR image to simulate the LR image
        lr_image = F.interpolate(hr_image, scale_factor=1 / self.scale_factor, mode='bicubic', align_corners=False)

        # Step 2: Upsample the LR image back to the original size (SR target size) using bicubic interpolation
        bicubic_sr_image = F.interpolate(lr_image, size=hr_image.shape[-2:], mode='bicubic', align_corners=False)

        # Step 3: Bicubic downsample the generated HR image to simulate its LR image
        lr_sr_image = F.interpolate(sr_image, scale_factor=1 / self.scale_factor, mode='bicubic', align_corners=False)

        # Step 4: Apply bicubic interpolation to the degraded SR image generated by the model
        bicubic_sr_image_model = F.interpolate(lr_sr_image, size=hr_image.shape[-2:], mode='bicubic', align_corners=False)

        # Step 5: Calculate the loss between the bicubic-interpolated SR image from the model and the bicubic SR image
        loss = self.loss_fn(bicubic_sr_image_model, bicubic_sr_image)
        return loss
