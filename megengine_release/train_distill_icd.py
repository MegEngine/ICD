# -*- coding: utf-8 -*-
# This repo is licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import bisect
import copy
import os
import time

import megengine as mge
import megengine.distributed as dist
from megengine.autodiff import GradManager
from megengine.data import DataLoader, Infinite, RandomSampler
from megengine.data import transform as T
from megengine.optimizer import AdamW, SGD
import math

from megengine.core._imperative_rt.utils import _set_defrag
_set_defrag(True)

from layers.tools.data_mapper import data_mapper
from layers.tools.utils import (
    AverageMeter,
    DetectionPadCollator,
    GroupedRandomSampler,
    get_config_info,
    import_from_file
)

logger = mge.get_logger(__name__)
logger.setLevel("INFO")
mge.device.set_prealloc_config(1024, 1024, 256 * 1024 * 1024, 4.0)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", default="net.py", type=str, help="net description file"
    )

    parser.add_argument(
        "-df", "--distill_file", default="distill_configs/ICD.py", type=str, help="distill description file"
    )

    parser.add_argument(
        "-tf", "--teacher_file", default="net.py", type=str, help="net description file"
    )

    parser.add_argument(
        "-w", "--weight_file", default=None, type=str, help="weights file",
    )

    parser.add_argument(
        "-tw", "--teacher_weight_file", default=None, type=str, help="weights file",
    )

    parser.add_argument(
        "-n", "--devices", default=1, type=int, help="total number of gpus for training",
    )
    parser.add_argument(
        "-b", "--batch_size", default=2, type=int, help="batch size for training",
    )
    parser.add_argument(
        "-d", "--dataset_dir", default="/data/datasets", type=str,
    )

    parser.add_argument(
        "-l", "--load_head", action='store_true'
    )

    parser.add_argument(
        "-sp", "--save_path", type=str,
    )


    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    # ------------------------ begin training -------------------------- #
    logger.info("Device Count = %d", args.devices)

    from datetime import datetime
    now = datetime.now()

    log_dir = "log-of-ICD-{}-{}".format(os.path.basename(args.file).split(".")[0], now.strftime('%H:%M') + ('-Inherent' if args.load_head else ''))
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    args.save_path = log_dir

    if args.devices > 1:
        trainer = dist.launcher(worker, n_gpus=args.devices)
        trainer(args)
    else:
        worker(args)


def worker(args):
    mge.dtr.eviction_threshold = "6GB" # set the eviction threshold to 5 GB
    mge.dtr.enable()                   # enable the DTR optimization

    current_network = import_from_file(args.file)

    model = current_network.Net(current_network.Cfg())
    model.train()

    ########### BUILD TEACHER MODEL ############
    current_teacher = import_from_file(args.teacher_file)
    cfg_tea = current_teacher.Cfg()
    cfg_tea.backbone_pretrained = False
    model_tea = current_teacher.Net(cfg_tea)
    model_tea.train()
    # model_tea.training = True # run in train mod

    state_dict_tea = mge.load(args.teacher_weight_file)
    if "state_dict" in state_dict_tea:
        state_dict_tea = state_dict_tea["state_dict"]
    model_tea.load_state_dict(state_dict_tea)

    ############### LOADED ####################

    ################## DISTILLER ##############
    distiller_cfg = import_from_file(args.distill_file)
    distiller = distiller_cfg.Net()
    ############### END ####################

    if dist.get_rank() == 0:
        logger.info(get_config_info(model.cfg))
        logger.info(repr(model))

    params_with_grad = []
    for name, param in model.named_parameters():
        if "bottom_up.conv1" in name and model.cfg.backbone_freeze_at >= 1:
            continue
        if "bottom_up.layer1" in name and model.cfg.backbone_freeze_at >= 2:
            continue
        params_with_grad.append(param)

    opt = SGD(
        params_with_grad,
        lr=model.cfg.basic_lr * args.batch_size * dist.get_world_size(),
        momentum=model.cfg.momentum,
        weight_decay=model.cfg.weight_decay,
    )

    opt_d = AdamW(
        distiller.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )
    params_with_grad.extend(distiller.parameters())
    gm = GradManager()
    if dist.get_world_size() > 1:
        gm.attach(
            params_with_grad,
            callbacks=[dist.make_allreduce_cb("mean", dist.WORLD)]
        )
    else:
        gm.attach(params_with_grad)

    if args.weight_file is not None:
        weights = mge.load(args.weight_file)
        model.backbone.bottom_up.load_state_dict(weights, strict=False)

    if args.load_head:
        print('Loading Prameters Besides from Backbones.')
        res = model.load_state_dict({k:v for k, v in model_tea.state_dict().items() if 'bottom_up' not in k}, strict=False)

    if dist.get_world_size() > 1:
        dist.bcast_list_(model.parameters())  # sync parameters
        dist.bcast_list_(distiller.parameters())  # sync parameters
        dist.bcast_list_(model.buffers())  # sync buffers
        dist.bcast_list_(distiller.buffers())  # sync parameters

    if dist.get_rank() == 0:
        logger.info("Prepare dataset")
    train_loader = iter(build_dataloader(args.batch_size, args.dataset_dir, model.cfg))

    ############## REGISTER ###############
    # To get intermediate representations
    holder = [None, None]
    def register_tea_hooker(m, i, o):
        holder[0] = o
        return

    model_tea.backbone.register_forward_hook(register_tea_hooker)

    def register_stu_hooker(m, i, o):
        holder[1] = o
        return

    model.backbone.register_forward_hook(register_stu_hooker)

    

    for epoch in range(model.cfg.max_epoch):
        train_one_epoch(model, model_tea, distiller, holder, train_loader, opt, opt_d, gm, epoch, args)
        if dist.get_rank() == 0:
            save_path = "{}/epoch_{}.pkl".format(
                args.save_path, epoch
            )
            mge.save(
                {"epoch": epoch, "state_dict": model.state_dict()}, save_path,
            )
            logger.info("dump weights to %s", save_path)


def train_one_epoch(model, model_tea, distiller, feat_holder, data_queue, opt, opt_d, gm, epoch, args):
    
    def train_func(image, im_info, gt_boxes, **args):
        model_tea(image=image, im_info=im_info, gt_boxes=gt_boxes)
        with gm:
            loss_dict = model(image=image, im_info=im_info, gt_boxes=gt_boxes)
            loss_distill = distiller(feat_holder[0], feat_holder[1], image, gt_boxes, im_info, distill_flag=0 if args['cur_step'] < 1000 else 1)
            loss_dict.update(loss_distill)
            loss_dict["total_loss"] = loss_dict["total_loss"] + sum(loss_distill.values())

            gm.backward(loss_dict["total_loss"])

            loss_list = list(loss_dict.values())

        mge.optimizer.clip_grad_norm(
                        distiller.parameters(), 0.01)
        opt.step().clear_grad()
        opt_d.step().clear_grad()
        return loss_list

    meter = AverageMeter(record_len=model.cfg.num_losses + distiller.num_losses)
    time_meter = AverageMeter(record_len=2)
    log_interval = model.cfg.log_interval
    tot_step = model.cfg.nr_images_epoch // (args.batch_size * dist.get_world_size())
    full_step = tot_step * model.cfg.max_epoch
    
    for step in range(tot_step):
        cur_step = tot_step * epoch + step
        adjust_learning_rate(opt, epoch, step, model.cfg, args)
        adjust_learning_rate_cos(opt_d, cur_step, full_step)

        data_tik = time.time()
        mini_batch = next(data_queue)
        data_tok = time.time()

        tik = time.time()
        loss_list = train_func(
            image=mge.tensor(mini_batch["data"]),
            im_info=mge.tensor(mini_batch["im_info"]),
            gt_boxes=mge.tensor(mini_batch["gt_boxes"]),
            cur_step=cur_step,
            full_step=full_step
        )

        tok = time.time()

        time_meter.update([tok - tik, data_tok - data_tik])

        if dist.get_rank() == 0:
            info_str = "e%d, %d/%d, lr:%f, "
            loss_str = ", ".join(
                ["{}:%f".format(loss) for loss in model.cfg.losses_keys] + ["{}:%f".format(loss) for loss in distiller.loss_keys]
            )
            time_str = ", train_time:%.3fs, data_time:%.3fs"
            log_info_str = info_str + loss_str + time_str
            meter.update([loss.numpy() for loss in loss_list])
            if step % log_interval == 0:
                logger.info(
                    log_info_str,
                    epoch,
                    step,
                    tot_step,
                    opt.param_groups[0]["lr"],
                    *meter.average(),
                    *time_meter.average()
                )
                meter.reset()
                time_meter.reset()


def adjust_learning_rate_cos(optimizer, cur_iter, total_iter):
    base_lr = 1e-4
    # Warm up
    lr = 0.5 * base_lr * (1 + math.cos(cur_iter / total_iter * math.pi))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    

def adjust_learning_rate(optimizer, epoch, step, cfg, args):
    base_lr = (
        cfg.basic_lr * args.batch_size * dist.get_world_size() * (
            cfg.lr_decay_rate
            ** bisect.bisect_right(cfg.lr_decay_stages, epoch)
        )
    )
    # Warm up
    lr_factor = 1.0
    if epoch == 0 and step < cfg.warm_iters:
        lr_factor = (step + 1.0) / cfg.warm_iters
    for param_group in optimizer.param_groups:
        param_group["lr"] = base_lr * lr_factor


def build_dataset(dataset_dir, cfg):
    data_cfg = copy.deepcopy(cfg.train_dataset)
    data_name = data_cfg.pop("name")

    data_cfg["root"] = os.path.join(dataset_dir, data_name, data_cfg["root"])

    if "ann_file" in data_cfg:
        data_cfg["ann_file"] = os.path.join(dataset_dir, data_name, data_cfg["ann_file"])

    data_cfg["order"] = ["image", "boxes", "boxes_category", "info"]

    return data_mapper[data_name](**data_cfg)


# pylint: disable=dangerous-default-value
def build_sampler(train_dataset, batch_size, aspect_grouping=[1]):
    def _compute_aspect_ratios(dataset):
        aspect_ratios = []
        for i in range(len(dataset)):
            info = dataset.get_img_info(i)
            aspect_ratios.append(info["height"] / info["width"])
        return aspect_ratios

    def _quantize(x, bins):
        return list(map(lambda y: bisect.bisect_right(sorted(bins), y), x))

    if len(aspect_grouping) == 0:
        return Infinite(RandomSampler(train_dataset, batch_size, drop_last=True))

    aspect_ratios = _compute_aspect_ratios(train_dataset)
    group_ids = _quantize(aspect_ratios, aspect_grouping)
    return Infinite(GroupedRandomSampler(train_dataset, batch_size, group_ids))


def build_dataloader(batch_size, dataset_dir, cfg):
    train_dataset = build_dataset(dataset_dir, cfg)
    train_sampler = build_sampler(train_dataset, batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        transform=T.Compose(
            transforms=[
                T.ShortestEdgeResize(
                    cfg.train_image_short_size,
                    cfg.train_image_max_size,
                    sample_style="choice",
                ),
                T.RandomHorizontalFlip(),
                T.ToMode(),
            ],
            order=["image", "boxes", "boxes_category"],
        ),
        collator=DetectionPadCollator(),
        num_workers=2,
    )
    return train_dataloader


if __name__ == "__main__":
    main()
