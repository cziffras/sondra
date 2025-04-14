import torch
import torch.nn.functional as F
from torchcvnn.datasets import PolSFDataset, ALOSDataset
import logging
import random
from torchcvnn.transforms import LogAmplitude, PolSARtoTensor
from ..transforms import SARContrastiveAugmentations
from ..utils import ToTensor
import torchvision.transforms.v2 as v2
import pathlib


class PolSFContrastive(ALOSDataset):
    def __init__(self, rootdir: str, transform_contrastive=None, **kwargs):
        root = pathlib.Path(rootdir) / "VOL-ALOS2044980750-150324-HBQR1.1__A"
        super().__init__(root, transform=None, **kwargs)
        self.transform_contrastive = transform_contrastive

    def __getitem__(self, index: int):
        data = super().__getitem__(index)

        if self.transform_contrastive is not None:
            view1, view2 = self.transform_contrastive(data)
        else:
            view1 = data
            view2 = data

        return (view1.to(torch.complex64), view2.to(torch.complex64))


class WrappedPolSFDataset(PolSFDataset):
    # labels to torch tensor
    to_tensor_labels = ToTensor(dtype=torch.int64)

    def __init__(self, root, transform=None, patch_size=(128, 128), patch_stride=None):
        super().__init__(
            root=root,
            transform=transform,
            patch_size=patch_size,
            patch_stride=patch_stride,
        )
        self._augment_transform = None

    def set_augment_transform(self, transform):
        self._augment_transform = transform

    def __getitem__(self, idx):
        data, labels = super().__getitem__(idx)
        labels = self.to_tensor_labels(labels)
        if self._augment_transform is not None:
            data = self._augment_transform(data)
        return data, labels


def get_polsf_contrastive_dataset(
    rootdir,
    patch_size=(128, 128),
    patch_stride=None,
    transform_contrastive=None,
):
    """
    Labeled pixels between corners : crop_coordinates = ((2832, 736), (7888, 3520)) (cf. torchcvnn.datasets.PolSFDataset classs)
    Image corners : ((0, 0), (22608, 8080))

    So to get the contrastive pretraining unlabled pixels, we concatenate datasets on following crop_coordinates :

    (0, 0), (7888, 736) (bottom left corner)
    (0, 736), (2832, 8080) (left side)
    (7888, 0),(22608, 8080) (right side)
    (2832, 3520), (7888, 8080) (remaining top side)
    """

    crop_coordinates_list = [
        ((0, 0), (7888, 736)),
        ((0, 736), (2832, 8080)),
        ((7888, 0), (22608, 8080)),
        ((2832, 3520), (7888, 8080)),
    ]

    complete_daset = torch.utils.data.ConcatDataset(
        [
            PolSFContrastive(
                rootdir=rootdir,
                patch_size=patch_size,
                patch_stride=patch_stride,
                transform_contrastive=transform_contrastive,
                crop_coordinates=crop_coordinates,
            )
            for crop_coordinates in crop_coordinates_list
        ]
    )
    return complete_daset


def get_transform(data_config):
    transform_args = data_config.get("transform", {})
    transform_name = transform_args.get("name", "SARContrastiveAugmentations")
    transform_params = transform_args.get("params", {})

    return eval(f"{transform_name}")(**transform_params)


def get_polsf_dataloaders(
    data_config,
    use_cuda,
    contrastive=False,
    debug=False,
):
    """
    Returns the train and validation dataloaders for the PolSF dataset,
    as well the list of patches part of test set.

    data_config : "data" section of the configuration file.
    use_cuda: bool, set True when a graphic card is available.
    """
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    root_dir = data_config["root_dir"]

    patch_size = tuple(data_config.get("patch_size", (128, 128)))
    patch_stride = tuple(data_config.get("patch_stride", patch_size))

    if contrastive:
        transform_contrastive_args = data_config.get("transform_contrastive", {})

        transform_name = transform_contrastive_args.get(
            "name", "SARContrastiveAugmentations"
        )
        transform_params = transform_contrastive_args.get("params", {})

        transform = eval(f"{transform_name}")(**transform_params)

        polsf_dataset = PolSFContrastive(
            rootdir=root_dir,
            patch_size=patch_size,
            patch_stride=patch_stride,
            transform_contrastive=v2.Compose(
                [PolSARtoTensor(), transform, LogAmplitude()]
            ),
        )

    else:
        polsf_dataset = WrappedPolSFDataset(
            root=root_dir,
            patch_size=patch_size,
            patch_stride=patch_stride,
            transform=v2.Compose([PolSARtoTensor()]),
        )

    if debug:
        # ========== DEBUG INFO ========== #
        print("\n========== Dataset Debug Info ==========")
        try:
            # Tentative d'accès au premier patch
            first_item = polsf_dataset[0]
            if isinstance(first_item, tuple):
                patch = first_item[0]  # Pour segmentation : (x, y)
            else:
                patch = first_item  # Pour contrastif : (x1, x2)

            print("Shape d’un patch extrait     :", patch.shape)
            print("Patch size (config)         :", patch_size)
            print("Patch stride (config)       :", patch_stride)
            print(f"Nombre total de patches     : {len(polsf_dataset)}")

        except Exception as e:
            print("Erreur pendant l'accès à un patch :", e)

        print("========================================\n")

    logging.info(f"  - I loaded {len(polsf_dataset)} samples")

    # Shuffle indices to create train, val and test sets
    indices = list(range(len(polsf_dataset)))
    random.shuffle(indices)
    num_valid = int(valid_ratio * len(polsf_dataset))

    valid_indices = indices[:num_valid]
    train_indices = indices[num_valid:]

    train_dataset = torch.utils.data.Subset(polsf_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(polsf_dataset, valid_indices)

    if not contrastive:
        # we are using the wrapped dataset
        transform = get_transform(data_config)
        train_dataset.dataset.set_augment_transform(
            v2.Compose([transform, LogAmplitude()])
        )
        valid_dataset.dataset.set_augment_transform(LogAmplitude())

    # Build the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    num_classes = 1
    if not contrastive:
        num_classes = len(polsf_dataset.classes)
    input_size = tuple(polsf_dataset[0][0].shape)

    return train_loader, valid_loader, input_size, num_classes

class GenericDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        """
        A generic dataset wrapper that works with any dataset class.

        Args:
            dataset: An instance of a dataset class (e.g., CIFAR10, MNIST, etc.).
        """
        self.dataset = dataset

    def __getitem__(self, index):
        """
        Fetch an item from the dataset.

        Args:
            index: Index of the item to fetch.

        Returns:
            A tuple containing (data, target, index).
        """
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            Length of the dataset.
        """
        return len(self.dataset)


def get_full_image_dataloader(data_config: dict) -> tuple:

    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    root_dir = data_config["root_dir"]

    patch_size = tuple(data_config.get("patch_size", (128, 128)))
    patch_stride = tuple(data_config.get("patch_stride", patch_size))
    img_size = patch_size[0]

    nsamples_per_cols, nsamples_per_rows = None, None

    base_dataset = WrappedPolSFDataset(
        root=root_dir,
        patch_size=patch_size,
        patch_stride=patch_stride,
        transform=v2.Compose([PolSARtoTensor(), LogAmplitude()]),
    )
    nsamples_per_cols = base_dataset.alos_dataset.nsamples_per_cols
    nsamples_per_rows = base_dataset.alos_dataset.nsamples_per_rows

    wrapped_dataset = GenericDatasetWrapper(base_dataset)

    data_loader = torch.utils.data.DataLoader(
        wrapped_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return (
        data_loader,
        nsamples_per_cols,
        nsamples_per_rows,
    )
