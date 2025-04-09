import sys
from .PolSF import get_polsf_dataloaders

def get_dataloaders(data_config, use_cuda, contrastive=False):
    """
    Calls function get_[dataset]_dataloaders for dataset given in config file.
    """
   
    module = sys.modules[__name__]
    func_name = f"get_{data_config['dataset']}_dataloaders"
    func = getattr(module, func_name)
    if contrastive:
        return func(data_config, use_cuda, contrastive=True)
    else:
        return func(data_config, use_cuda)