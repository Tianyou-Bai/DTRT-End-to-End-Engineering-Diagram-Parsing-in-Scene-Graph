# 分布式辅助函数
def is_dist_available_and_initialized():
    import torch.distributed as dist
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    import torch.distributed as dist
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()
"""
reference
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
- https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import os
import random
import numpy as np 
import atexit
import torch
import torch.nn as nn 
import torch.distributed
import torch.backends.cudnn
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader 
def setup_distributed(print_rank: int=0, print_method: str='builtin', seed: int=None, ):
    try:
        RANK = int(os.getenv('RANK', -1))
        LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  
        WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
        torch.distributed.init_process_group(init_method='env://')
        torch.distributed.barrier()
        rank = torch.distributed.get_rank()
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
        enabled_dist = True
        print('Initialized distributed mode...')
    except:
        enabled_dist = False
        print('Not init distributed mode.')
    setup_print(get_rank() == print_rank, method=print_method)
    if seed is not None:
        setup_seed(seed)
    return enabled_dist
def setup_print(is_main, method='builtin'):
    import builtins as __builtin__
    if method == 'builtin':
        builtin_print = __builtin__.print
    elif method == 'rich':
        import rich 
        builtin_print = rich.print
    else:
        raise AttributeError('')
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    atexit.register(lambda: torch.backends.cudnn.benchmark == True)
