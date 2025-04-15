from .PolSF_inprogress import PolSFDataManager

def get_polsf_dataloaders(data_config, use_cuda, contrastive=False):
    """
    Wrapper to call and instanciate get_dataloaders from PolSFDataManager
    """
    manager = PolSFDataManager(config=data_config, use_cuda=use_cuda, debug=data_config.get("debug", False))
    return manager.get_dataloaders(contrastive=contrastive)

def get_full_image_dataloaders(data_config, use_cuda, contrastive=False):
    """
    Wrapper to call and instanciate PolSFDataManager
    """
    manager = PolSFDataManager(config=data_config, use_cuda=use_cuda, debug=data_config.get("debug", False))
    return manager.get_full_image_dataloader(contrastive=contrastive)