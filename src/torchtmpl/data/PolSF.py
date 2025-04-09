import torch
import torch.nn.functional as F
from torchcvnn.datasets import PolSFDataset, ALOSDataset
import logging
import random
from ..transforms import *
from ..utils import ToTensor, PolSARtoTensor
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

    def __getitem__(self, idx):
        data, labels = super().__getitem__(idx)
        labels = self.to_tensor_labels(labels)
        return data, labels
    

def create_polsf_dataset(
    root_dir,
    patch_size=(128, 128),
    patch_stride=None,
    contrastive=False,
    transform_config=None,
    debug=False
):
    """
    Returns a PolSF dataset either for contrastive training or a supervised segmentation task.

    Labeled pixels between corners : 

    crop_coordinates = ((2832, 736), (7888, 3520)) (cf. torchcvnn.datasets.PolSFDataset classs)
    Image corners : ((0, 0), (22608, 8080))

    So to get the contrastive pretraining unlabled pixels, we concatenate datasets on 
    following crop_coordinates :

    (0, 0), (7888, 736) (bottom left corner)
    (0, 736), (2832, 8080) (left side)
    (7888, 0),(22608, 8080) (right side)
    (2832, 3520), (7888, 8080) (remaining top side)
    
    Args:
        root_dir
        patch_size
        patch_stride
        contrastive
        transform_config
        debug
    """
    patch_stride = patch_stride or patch_size
    
    if transform_config is None:
        transform_config = {}
    
    if contrastive:
        transform_name = transform_config.get("name", "SARContrastiveAugmentations")
        transform_params = transform_config.get("params", {})
        transform = eval(f"{transform_name}")(**transform_params)
     
        crop_coordinates_list = [
            ((0, 0), (7888, 736)),
            ((0, 736), (2832, 8080)),
            ((7888, 0), (22608, 8080)),
            ((2832, 3520), (7888, 8080)),
        ]
        
        polsf_dataset = torch.utils.data.ConcatDataset(
            [
                PolSFContrastive(
                    rootdir=root_dir,
                    patch_size=patch_size,
                    patch_stride=patch_stride,
                    transform_contrastive=v2.Compose([PolSARtoTensor(), transform]),
                    crop_coordinates=crop_coordinates,
                )
                for crop_coordinates in crop_coordinates_list
            ]
        )
    else:
        # Semantic segmentation
        transform_name = transform_config.get("name", None)
        transform_params = transform_config.get("params", {})
        transform = eval(f"{transform_name}")(**transform_params)
        
        polsf_dataset = WrappedPolSFDataset(
            root=root_dir,
            patch_size=patch_size,
            patch_stride=patch_stride,
            transform=v2.Compose([PolSARtoTensor(), transform]),
        )

    if debug:
        print("\n========== Dataset Debug Info ==========")
        try:
            first_item = polsf_dataset[0]
            if isinstance(first_item, tuple):
                patch = first_item[0]  # Pour segmentation : (x, y)
            else:
                patch = first_item  # Pour contrastif : (x1, x2)

            print("Shape d'un patch extrait     :", patch.shape)
            print("Patch size (config)         :", patch_size)
            print("Patch stride (config)       :", patch_stride)
            print(f"Nombre total de patches     : {len(polsf_dataset)}")
        except Exception as e:
            print("Erreur pendant l'accès à un patch :", e)
        print("========================================\n")
        
    logging.info(f"  - I loaded {len(polsf_dataset)} samples")
    
    return polsf_dataset


def get_polsf_dataloaders(data_config, use_cuda, contrastive=False, debug=False):
    """
    Creates PolSF dataloaders.
    
    Args:
        data_config: Section "data" section of config file
        use_cuda
        contrastive
        debug: if True yields debugging informations
        
    Returns:
        train_loader, valid_loader, input_size, num_classes
    """
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    root_dir = data_config["root_dir"]
    patch_size = tuple(data_config.get("patch_size", (128, 128)))
    patch_stride = tuple(data_config.get("patch_stride", patch_size))
    
    if contrastive:
        transform_config = data_config.get("transform_contrastive", {})
    else:
        transform_config = data_config.get("transform", {})
    
    polsf_dataset = create_polsf_dataset(
        root_dir=root_dir,
        patch_size=patch_size,
        patch_stride=patch_stride,
        contrastive=contrastive,
        transform_config=transform_config,
        debug=debug
    )

    indices = list(range(len(polsf_dataset)))
    random.shuffle(indices)
    num_valid = int(valid_ratio * len(polsf_dataset))

    valid_indices = indices[:num_valid]
    train_indices = indices[num_valid:]

    train_dataset = torch.utils.data.Subset(polsf_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(polsf_dataset, valid_indices)

    # dataloaders
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

    # Determining num classes and size of input 
    num_classes = 1
    if not contrastive:
        num_classes = len(polsf_dataset.classes)
    
    sample = polsf_dataset[0]
    if isinstance(sample, tuple):
        input_size = tuple(sample[0].shape)  # segmentation: (x, y)
    else:
        input_size = tuple(sample[0].shape)  # contrastive: (x1, x2)

    return train_loader, valid_loader, input_size, num_classes


if __name__ == "__main__":

    dataset = create_polsf_dataset(
        rootdir="datasets/Polarimetric-SanFrancisco/SAN_FRANCISCO_ALOS2/"
    )
    print(f"concat len : {len(dataset)}")
