import torch
import random
import logging
import pathlib
from torchcvnn.datasets import ALOSDataset
from torchcvnn.transforms import LogAmplitude, PolSARtoTensor
import torchvision.transforms.v2 as v2
import numpy as np
from PIL import Image

from ..utils import ToTensor
from ..transforms import SARContrastiveAugmentations


class EnhancedPolSFDataset(ALOSDataset):
    """
    Dataset hybride : en mode supervisé (contrastive_mode=False), 
    le dataset se comporte comme PolSFDataset en fournissant patchs et labels,
    tandis qu'en mode contrastif il se comporte comme ALOSDataset avec 
    des transformations contrastives.
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
        crop_coordinates=None,
        patch_stride=None,
        index_tracking=False,
        **kwargs
    ):
        # For supervised mode force crop coords
        if not contrastive_mode:
            crop_coordinates = ((2832, 736), (7888, 3520))

        if isinstance(root, str) and not root.endswith(self.ALOS_PATH_SUFFIX):
            root = pathlib.Path(root) / self.ALOS_PATH_SUFFIX
        
        super().__init__(
            volpath=root,
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

        self.classes = [
                "0 - unlabel",
                "1 - Montain",
                "2 - Water",
                "3 - Vegetation",
                "4 - High-Density Urban",
                "5 - Low-Density Urban",
                "6 - Developd",
            ]
        
        if not contrastive_mode:
            labels_path = root.parent / "SF-ALOS2-label2d.png"
            self.labels = np.array(Image.open(labels_path))[::-1, :].copy()  

    def __getitem__(self, idx):
        if self.contrastive_mode:
            return self._get_contrastive_item(idx)
        else:
            return self._get_supervised_item(idx)
    
    def _get_contrastive_item(self, idx):

        data = super().__getitem__(idx)
        
        if self.transform_contrastive is not None:
            view1, view2 = self.transform_contrastive(data)
        else:
            view1, view2 = data, data
        
        view1 = view1.to(torch.complex64)
        view2 = view2.to(torch.complex64)
        
        if self.index_tracking:
            return view1, view2, idx
        return view1, view2

    def _get_supervised_item(self, idx):
        """
        Supervised mode : get the patch and corresponding label behaving like PolSFDataset.
        """
        # Get the patch from ALOS Dataset
        patch = super().__getitem__(idx)
        
        row_stride, col_stride = self.patch_stride 
        nsamples_per_cols = self.nsamples_per_cols
        
        start_row = (idx // nsamples_per_cols) * row_stride
        start_col = (idx % nsamples_per_cols) * col_stride
        
        num_rows, num_cols = self.patch_size
        labels = self.labels[
            start_row: (start_row + num_rows), start_col: (start_col + num_cols)
        ]

        # WARNING : augmentation transform is NOT a contrastive transform
        if self.augment_transform is not None:
            patch = self.augment_transform(patch)

        labels = self.to_tensor_labels(labels)
        
        if self.index_tracking:
            return patch, labels, idx
        return patch, labels


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
        
        transform = self._get_transform(contrastive=True)
     
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
            transform=v2.Compose([PolSARtoTensor(), LogAmplitude()]),
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
    
    def _get_transform(self, contrastive=False):
        """Get transform from config (see transforms directory to see what options we got)"""
        
        if contrastive:
            transform_args = self.config.get("transform_contrastive", {})
            transform_params = transform_args.get("params", {})
            transform = SARContrastiveAugmentations(**transform_params)
        else:
            transform_args = self.config.get("transform_supervised", {})
            transform_params = transform_args.get("params", {})
            transform = lambda x: x  # KEY FEATURE TO IMPLEMENT : SARAugmentations(**transform_params)
        
        return transform
    
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
    import sys
    import yaml
    from collections import defaultdict

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 2:
        logging.error(f"Usage: {sys.argv[0]} config.yaml")
        sys.exit(-1)

    config_file = sys.argv[1]

    logging.info(f"Loading config from {config_file}")
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Erreur lors du chargement du fichier de config: {e}")
        sys.exit(-1)

    data_config = config["data"]
    data_manager = PolSFDataManager(data_config, use_cuda=True, debug=True)

    if data_config.get("contrastive", False):
        train_loader_c, val_loader_c, input_size_c, num_classes_c = data_manager.get_dataloaders(contrastive=True)
    
        print("\n=== Contrastive mode ===")
        print(f"Num samples in train loader (contrastive) : {len(train_loader_c.dataset)}")
        print(f"Num samples in val loader (contrastive)   : {len(val_loader_c.dataset)}")
        print(f"Input size (contrastive) : {input_size_c}")
        print(f"Num classes (contrastive) : {num_classes_c}, should be 1")

    else: 
        train_loader, val_loader, input_size, num_classes = data_manager.get_dataloaders(contrastive=False)
        
        print("=== Supervised mode (Standard) ===")
        print(f"Num samples in train loader : {len(train_loader.dataset)}")
        print(f"Num samples in val loader : {len(val_loader.dataset)}")
        print(f"Input size : {input_size}")
        print(f"Num classes : {num_classes}")
        
    if config.get("visualize", False):
        full_loader, cols, rows = data_manager.get_full_image_dataloader()
        print("\n=== Full Image Dataloader ===")
        print(f"Num samples in full loader : {len(full_loader.dataset)}")
        print(f"Num cols patches : {cols}")
        print(f"Num rows patches : {rows}")
    
    try:
        train_loader, val_loader, input_size, num_classes = data_manager.get_dataloaders(contrastive=False)
        std_batch = next(iter(train_loader))
        print("\n== Batch in supervised mode ==")
        if isinstance(std_batch, (list, tuple)):
            data_std = std_batch[0]
            target_std = std_batch[1]
            print(f"Shape of input : {data_std.shape}, dtype : {data_std.dtype}")
            print(f"Shape of target : {target_std.shape}, dtype : {target_std.dtype}")

            loader = train_loader

            total_targets = 0
            total_uniform = 0
            for batch in loader:
                data, targets = batch
                
                for target in targets:
                    total_targets += 1
                    if torch.all(target == target[0, 0]).item():
                        total_uniform += 1

            print("== Evaluating uniformity in targets ==")
            print(f"{total_targets/total_uniform:.2f}% of targets are uniform")

            classes_count = defaultdict(int)

            max_class = 7 
            bins_total = None

            for batch in loader:
                _, targets = batch
                flattened_targets = targets.view(-1)
                bins = torch.bincount(flattened_targets, minlength=max_class)
                if bins_total is None:
                    bins_total = bins.clone()
                else:
                    bins_total += bins

            total_occurrences = torch.sum(bins_total).item()

            print("== Evaluation classes repartition in targets ==")
            for i, count in enumerate(bins_total.tolist()):
                percentage = count / total_occurrences * 100
                print(f"Classe {i} : {percentage:.2f}%")

        else:
            print(f"Type : {type(std_batch)}")
    except Exception as e:
        print(f"Error trying to access batch : {e}")
    
    try:
        full_loader, cols, rows = data_manager.get_full_image_dataloader()
        full_batch = next(iter(full_loader))
        print("\n== Batch in full image dataloader ==")
        if isinstance(full_batch, (list, tuple)):
            data_full = full_batch[0]
            print(f"Shape : {data_full.shape}")
        else:
            print(f"Type : {type(full_batch)}")
    except Exception as e:
        print(f"Error trying to access batch full image: {e}")