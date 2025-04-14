import torch
import random
import logging
import pathlib
from torchcvnn.datasets import PolSFDataset, ALOSDataset
from torchcvnn.transforms import LogAmplitude, PolSARtoTensor
import torchvision.transforms.v2 as v2

from ..utils import ToTensor
from ..transforms import SARContrastiveAugmentations


class EnhancedPolSFDataset(PolSFDataset):
    """
    PolSF Dataset but managing contrastive mode as much as configurable transforms...
    """
    ALOS_PATH_SUFFIX = "VOL-ALOS2044980750-150324-HBQR1.1__A"
    
    def __init__(
        self, 
        root, 
        contrastive_mode=False,
        transform=None,
        transform_contrastive=None,
        augment_transform=None,
        patch_size=(128, 128), 
        patch_stride=None,
        crop_coordinates=None,
        index_tracking=False,
        **kwargs
    ):
        
        if isinstance(root, str) and not root.endswith(self.ALOS_PATH_SUFFIX):
            root = pathlib.Path(root) / self.ALOS_PATH_SUFFIX
            
        super().__init__(
            root=root,
            transform=transform,
            patch_size=patch_size,
            patch_stride=patch_stride,
            crop_coordinates=crop_coordinates,
            **kwargs
        )
        
        self.contrastive_mode = contrastive_mode
        self.transform_contrastive = transform_contrastive
        self.augment_transform = augment_transform
        self.index_tracking = index_tracking
        self.to_tensor_labels = ToTensor(dtype=torch.int64)
        
    def __getitem__(self, idx):
        if self.contrastive_mode:
            return self._get_contrastive_item(idx)
        else:
            return self._get_standard_item(idx)
    
    def _get_contrastive_item(self, idx):
        data = super(ALOSDataset, self).__getitem__(idx)
        
        if self.transform_contrastive is not None:
            view1, view2 = self.transform_contrastive(data)
        else:
            view1, view2 = data, data
            
        result = (view1.to(torch.complex64), view2.to(torch.complex64))
        
        # Add the index if needed
        if self.index_tracking:
            return (*result, idx)
        return result
    
    def _get_standard_item(self, idx):
        """Standard get_item method from polsf"""
        data, labels = super().__getitem__(idx)
        
        # converting labels to tensor
        labels = self.to_tensor_labels(labels)
        
        # WARNING: augment transform does NOT play the role of a contrastive one
        if self.augment_transform is not None:
            data = self.augment_transform(data)
            
        result = (data, labels)
        
        # Add the index if needed
        if self.index_tracking:
            return (*result, idx)
        return result


class PolSFDataManager:
    """
    Manager that handles both datasets and dataloaders creation.
    """

    
    def __init__(self, config, use_cuda=False, debug=False):

        self.config = config
        self.use_cuda = use_cuda
        self.debug = debug
        
        self.root_dir = config["root_dir"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.valid_ratio = config["valid_ratio"]
        self.patch_size = tuple(config.get("patch_size", (128, 128)))
        self.patch_stride = tuple(config.get("patch_stride", self.patch_size))

    
    def get_dataloaders(self, contrastive=False):
    
        if contrastive:
            dataset = self._create_contrastive_dataset()
        else:
            dataset = self._create_standard_dataset()
        
        # Debug if needed
        if self.debug:
            self._print_debug_info(dataset)
        
        logging.info(f"Loaded {len(dataset)} samples")
        
        train_dataset, valid_dataset = self._split_dataset(dataset)
        
        if not contrastive:
            transform = self._get_transform()
            train_dataset = EnhancedPolSFDataset(
                root=self.root_dir,
                contrastive_mode=False,
                transform=v2.Compose([PolSARtoTensor()]),
                augment_transform=v2.Compose([transform, LogAmplitude()]),
                patch_size=self.patch_size,
                patch_stride=self.patch_stride
            )
            
            # Just apply LogAmplitude to valid dataset ("smoothing" technique)
            valid_dataset = EnhancedPolSFDataset(
                root=self.root_dir,
                contrastive_mode=False,
                transform=v2.Compose([PolSARtoTensor()]),
                augment_transform=LogAmplitude(),
                patch_size=self.patch_size,
                patch_stride=self.patch_stride
            )
        
        # Loaders are created here
        loader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.use_cuda,
        }
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, **loader_kwargs
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, shuffle=False, **loader_kwargs
        )
        
        num_classes = 1 if contrastive else len(dataset.classes)
        input_size = tuple(dataset[0][0].shape)
        
        return train_loader, valid_loader, input_size, num_classes
    
    
    def get_full_image_dataloader(self):

        dataset = EnhancedPolSFDataset(
            root=self.root_dir,
            contrastive_mode=False,
            transform=v2.Compose([PolSARtoTensor()]),
            augment_transform=LogAmplitude(),
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            index_tracking=True
        )
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        
        alos_dataset = dataset.alos_dataset if hasattr(dataset, 'alos_dataset') else dataset
        
        return (
            loader,
            alos_dataset.nsamples_per_cols,
            alos_dataset.nsamples_per_rows,
        )
    
    
    def _create_contrastive_dataset(self):

        # Selects unanottated regions of the total image to 
        # leverage contrastive pretraining
        crop_regions = [
            ((0, 0), (7888, 736)),        # bottom left
            ((0, 736), (2832, 8080)),     # left side
            ((7888, 0), (22608, 8080)),   # right side
            ((2832, 3520), (7888, 8080)), # top side
        ]
        
        transform_args = self.config.get("transform_contrastive", {})
        transform_params = transform_args.get("params", {})
        transform = SARContrastiveAugmentations(**transform_params)
     
        transform_pipeline = v2.Compose([PolSARtoTensor(), transform, LogAmplitude()])
        
    
        datasets = []
        for coords in crop_regions:
            datasets.append(
                EnhancedPolSFDataset(
                    root=self.root_dir,
                    contrastive_mode=True,
                    transform_contrastive=transform_pipeline,
                    patch_size=self.patch_size,
                    patch_stride=self.patch_stride,
                    crop_coordinates=coords,
                )
            )
        
        return torch.utils.data.ConcatDataset(datasets)
    
    
    def _create_standard_dataset(self):

        return EnhancedPolSFDataset(
            root=self.root_dir,
            contrastive_mode=False,
            transform=v2.Compose([PolSARtoTensor()]),
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
        )
    
    
    def _split_dataset(self, dataset):
        """Splits a dataset in train and valid subsets"""
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        split = int(self.valid_ratio * len(dataset))
        
        return (
            torch.utils.data.Subset(dataset, indices[split:]),
            torch.utils.data.Subset(dataset, indices[:split]),
        )
    
    def _get_transform(self):
        """Get transform from config (see transforms directory to see what options we got)"""
        transform_args = self.config.get("transform", {})
        transform_name = transform_args.get("name", "SARContrastiveAugmentations")
        transform_params = transform_args.get("params", {})
        
        transforms_dict = {
            "SARContrastiveAugmentations": SARContrastiveAugmentations,
        }
        
        if transform_name not in transforms_dict:
            raise ValueError(f"Transform not found: {transform_name}")
        
        return transforms_dict[transform_name](**transform_params)
    
    def _print_debug_info(self, dataset):
        print("\n========== Dataset Debug Info ==========")
        try:
            first_item = dataset[0]
            patch = first_item[0] if isinstance(first_item, tuple) else first_item
                
            print(f"Patch shape: {patch.shape}")
            print(f"Patch size: {self.patch_size}")
            print(f"Patch stride: {self.patch_stride}")
            print(f"Total patches: {len(dataset)}")
        except Exception as e:
            print(f"Error accessing patch: {e}")
        print("========================================\n")


if __name__ =="__main__":

    data_config = {
    "root_dir": "/path/to/data",
    "batch_size": 32,
    "num_workers": 4,
    "valid_ratio": 0.2,
    "patch_size": [128, 128],
    "transform": {
        "name": "SARContrastiveAugmentations",
        "params": {"p": 0.5}
    }
}
    data_manager = PolSFDataManager(data_config, use_cuda=True, debug=True)

    train_loader, val_loader, input_size, num_classes = data_manager.get_dataloaders(contrastive=False)

    train_loader_contrastive, val_loader_contrastive, input_size_c, num_classes_c = data_manager.get_dataloaders(contrastive=True)

    full_loader, cols, rows = data_manager.get_full_image_dataloader()