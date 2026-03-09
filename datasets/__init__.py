# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .pid_datasets import build_pid_dataset, CocoCompatibleDataset
# from .pid2graph_datasets import build_pid2graph  # 临时注释，文件为空


def get_coco_api_from_dataset(dataset):
    # Preserve subset indices to align image ids between predictions and COCO base
    subset_indices = None
    ds = dataset
    for _ in range(10):
        if isinstance(ds, torch.utils.data.Subset):
            if subset_indices is None:
                subset_indices = ds.indices
            ds = ds.dataset
        else:
            break
    if isinstance(ds, torchvision.datasets.CocoDetection):
        return ds.coco
    # Try to wrap custom dataset to COCO-compatible API with indices alignment
    try:
        return CocoCompatibleDataset(ds, indices=subset_indices).coco
    except Exception:
        return None


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'pid':
        return build_pid_dataset(image_set, args)
    # if args.dataset_file == 'pid2graph':
    #     return build_pid2graph(image_set, args)  # 临时注释
    raise ValueError(f'dataset {args.dataset_file} not supported')
