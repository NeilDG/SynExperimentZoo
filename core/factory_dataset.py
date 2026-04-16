import glob
import torch
import random
from loaders import superres_datasets, segmentation_datasets

class DatasetFactory:
    @staticmethod
    def get_dataloaders(config):
        modality = config.get('dataset.modality', 'image')
        
        if modality == 'image':
            return DatasetFactory._get_image_loaders(config)
        elif modality == 'video':
            return DatasetFactory._get_video_loaders(config)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    @staticmethod
    def _get_image_loaders(config):
        a_path = config.get('dataset.train.a_path')
        b_path = config.get('dataset.train.b_path')
        load_size = config.get('training.load_size', 1)
        num_workers = config.get('training.num_workers', 4)
        
        a_list = glob.glob(a_path) if a_path else []
        b_list = glob.glob(b_path) if b_path else []
        
        # Simple sample padding to reach ideal size if needed
        ideal_size = 100000
        if len(a_list) < ideal_size and len(a_list) > 0:
            a_list = (a_list * (ideal_size // len(a_list) + 1))[:ideal_size]
            b_list = (b_list * (ideal_size // len(b_list) + 1))[:ideal_size]

        train_dataset = superres_datasets.PairedImageDataset(a_list, b_list, 1)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=load_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True
        )

        # Test loader
        test_a = config.get('dataset.test.a_path')
        test_b = config.get('dataset.test.b_path')
        test_a_list = glob.glob(test_a) if test_a else []
        test_b_list = glob.glob(test_b) if test_b else []
        
        test_dataset = superres_datasets.PairedImageDataset(test_a_list, test_b_list, 2)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.get('training.test_batch_size', 1),
            num_workers=1
        )

        return train_loader, test_loader, len(a_list)

    @staticmethod
    def _get_video_loaders(config):
        """
        Placeholder for video loading logic.
        Standardizes input to [B, T, C, H, W].
        """
        print("Initializing video dataloaders...")
        # Implementation would use a VideoDataset that yields frames
        return None, None, 0
