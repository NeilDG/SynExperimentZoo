import torch
from model import vanilla_cycle_gan as cycle_gan, unet_gan, usi3d_gan, ffa_gan, rrdbnet, network_swinir, network_srmd
from model import attention_resunet_gan, densenet_gan, nafssr_gan, restormer_unet_gan, psp_net, translator_gan, new_style_transfer_gan, embedding_network

class ModelFactory:
    @staticmethod
    def create_model(config, device):
        """
        Creates generator and discriminator based on config.
        """
        model_type = config.get('model.type', 1)
        input_nc = config.get('model.input_nc', 3)
        num_blocks = config.get('model.num_blocks', 6)
        dropout_rate = config.get('model.dropout_rate', 0.0)
        norm_mode = config.get('model.norm_mode', 'batch')
        use_cbam = config.get('model.use_cbam', False)
        
        # Discriminator is usually CycleGAN-based for I2I tasks
        netD = cycle_gan.Discriminator(input_nc=3).to(device)

        if model_type == 1:
            netG = cycle_gan.Generator(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks, 
                                      dropout_rate=dropout_rate, use_cbam=use_cbam, norm=norm_mode).to(device)
        elif model_type == 2:
            netG = unet_gan.UnetGenerator(input_nc=input_nc, output_nc=3, num_downs=num_blocks).to(device)
        elif model_type == 3:
            params = {'dim': 64, 'mlp_dim': 256, 'style_dim': 8, 'n_layer': 3, 'activ': 'relu', 
                      'n_downsample': 2, 'n_res': num_blocks, 'pad_type': 'reflect'}
            netG = usi3d_gan.AdaINGen(input_dim=input_nc, output_dim=3, params=params).to(device)
        elif model_type == 4:
            netG = ffa_gan.FFABase(num_blocks, dropout_rate=dropout_rate).to(device)
        elif model_type == 5:
            netG = rrdbnet.RRDBNet(in_nc=input_nc, sf=1).to(device)
        elif model_type == 6:
            netG = network_swinir.SwinIR(upscale=1, window_size=8, img_range=1., depths=[6, 6, 6, 6],
                                        embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, 
                                        upsampler='pixelshuffledirect').to(device)
        elif model_type == 7:
            netG = attention_resunet_gan.Generator(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks,
                                                  dropout_rate=dropout_rate, norm=norm_mode).to(device)
        elif model_type == 8:
            netG = densenet_gan.Generator(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks).to(device)
        elif model_type == 9:
            netG = nafssr_gan.Generator(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks,
                                       dropout_rate=dropout_rate, norm=norm_mode).to(device)
        elif model_type == 10:
            netG = restormer_unet_gan.Generator(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks,
                                               dropout_rate=dropout_rate).to(device)
        elif model_type == 11:
            netG = psp_net.PSPNet(n_classes=3).to(device)
        elif model_type == 12:
            netG = translator_gan.Generator(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks).to(device)
        elif model_type == 13:
            netG = new_style_transfer_gan.Generator(nblocks=num_blocks).to(device)
        elif model_type == 14:
            netG = embedding_network.EmbeddingNetwork(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks).to(device)
        else:
            netG = network_srmd.SRMD(in_nc=input_nc, out_nc=3, nc=64, nb=num_blocks, upscale=1, 
                                    act_mode='R', upsample_mode='pixelshuffle').to(device)

        # Handle Video Wrapper if requested
        if config.get('dataset.modality') == 'video':
            netG = ModelFactory._wrap_video(netG, config)
            
        return netG, netD

    @staticmethod
    def _wrap_video(model, config):
        """
        Placeholder for wrapping a 2D model to handle video (temporal consistency).
        """
        print("Wrapping model for video modality...")
        # In a real scenario, this might add 3D convs or a temporal module
        return model
