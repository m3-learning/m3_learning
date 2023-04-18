import numpy as np

def random_seed(seed = 42, pytorch_ = True, numpy_ = True, tensorflow_ = True):
    """Function that sets the random seed

    Args:
        seed (int, optional): value for the seed. Defaults to 42.
        pytorch_ (bool, optional): chooses if you set the pytorch seed. Defaults to True.
        numpy_ (bool, optional): chooses if you set the numpy seed. Defaults to True.
        tensorflow_ (bool, optional): chooses if you set the tensorflow seed. Defaults to True.
    """    
    
    try:
        if pytorch_:
            import torch
            # torch.set_default_dtype(torch.float64)
            torch.manual_seed(seed)
            print(f'Pytorch seed was set to {42}')
    except: 
        pass
    
    try:
        np.random.seed(42)
        print(f'Numpy seed was set to {42}')
    except: 
        pass
    
    try: 
        import tensorflow as tf
        tf.random.set_seed(seed)
        print(f'tensorflow seed was set to {42}')
    except: 
        pass

