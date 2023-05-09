# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# import torch.utils.data
# import torchvision

from .hico import build as build_hico


# def get_coco_api_from_dataset(dataset):
#     for _ in range(10):
#         # if isinstance(dataset, torchvision.datasets.CocoDetection):
#         #     break
#         if isinstance(dataset, torch.utils.data.Subset):
#             dataset = dataset.dataset
#     if isinstance(dataset, torchvision.datasets.CocoDetection):
#         return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'hico':
        return build_hico(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
