# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
# import torch
import paddle
# from torch.utils.data import DataLoader, DistributedSampler
from paddle.io import DataLoader, DistributedBatchSampler,RandomSampler, SequenceSampler, BatchSampler
from paddle.optimizer import AdamW
from paddle.optimizer.lr import LRScheduler

import util.misc as utils
from datasets import build_dataset
from engine import  train_one_epoch, evaluate_hoi
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # HOI
    parser.add_argument('--hoi', action='store_true',
                        help="Train for HOI if the flag is provided")
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="Verb class coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--verb_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--hoi_path', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='gpu:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--model_name', default='baseline')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    # paddle.device.set_device(args.device)
    paddle.device.set_device("cpu")
    # device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    # model.to(device)

    model_without_ddp = model
    # todo
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     #  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.trainable)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.trainable]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.trainable],
            "lr": args.lr_backbone,
        },
    ]

    # for n, p in model_without_ddp.named_parameters():
    #     if p.trainable:
    #         print(" trained: ", n)
    #     else:
    #         print(" not trained: ", n)
    # assert False

    lr_scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.lr, step_size=args.lr_drop, gamma=0.1, verbose=True)
    # sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
    # AdamW(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, parameters=None, weight_decay=0.01, lr_ratio=None, apply_decay_param_fun=None, grad_clip=None, lazy_mode=False, multi_precision=False, name=None
    optimizer = AdamW(parameters=param_dicts, learning_rate=lr_scheduler,
                                  weight_decay=args.weight_decay)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
  
    if args.distributed:
        batch_sampler_train = DistributedBatchSampler(dataset=dataset_train, batch_size=args.batch_size, drop_last=True,shuffle=True)
        batch_sampler_val   = DistributedBatchSampler(dataset=dataset_val, batch_size=args.batch_size, drop_last=False,shuffle=False)
    else:
        batch_sampler_train = BatchSampler(dataset=dataset_train, batch_size=args.batch_size, drop_last=True,shuffle=True)
        batch_sampler_val   = BatchSampler(dataset=dataset_val, batch_size=args.batch_size, drop_last=False,shuffle=False)

    data_loader_train = DataLoader(dataset=dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset=dataset_val, batch_sampler=batch_sampler_val,
                                 collate_fn=utils.collate_fn, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = paddle.load(args.resume)
        model_without_ddp.load_state_dict(checkpoint['model'])

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    elif args.pretrained:
        checkpoint = paddle.load(args.pretrained)
        # model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        # model_without_ddp.load(args.pretrained, skip_mismatch=False, reset_optimizer=True)
        model_without_ddp.set_state_dict(checkpoint, use_structured_name=False)

        model_dict = {}
        for k, v in model_without_ddp.named_parameters():
            model_dict[k] = v

        for k in checkpoint['model']:
            if k in model_dict:
                if checkpoint['model'][k].shape != model_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, ' \
                          'loaded shape{}.'.format(
                        k, model_dict[k].shape, checkpoint['model'][k].shape))
                    checkpoint['model'][k] = model_dict[k]
                # else:
                #     print('load parameter {}.'.format(k))
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_dict:
            if not (k in checkpoint['model']):
                print('No param {}.'.format(k))

    if args.eval:
        if args.hoi:
            evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val,
                         args.subject_category_id)
            return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, epoch,
            args.clip_max_norm, args.model_name)
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        if args.hoi:
            test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val,
                                      args.subject_category_id)

            if 'mAP' in test_stats and float(test_stats["mAP"]) > 0.27:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, output_dir / f'checkpoint{epoch:04}.pth')


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
