#!/usr/bin/env python
# Codes contributed by Zijian Kang: Copyright (c) Megvii Inc. and its affiliates. All Rights Reserved
# Codes contributed by detectron2: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch import distributed
from torch.nn.parallel import DistributedDataParallel

from detectron2.structures import ImageList
from detectron2.solver.build import maybe_add_gradient_clipping
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from adet.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from utils.build import (
    my_build_optimizer,
    build_distill_configs,
    AllMetricPrinter
)
from detectron2.evaluation.evaluator import *
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

# import to register
import models.distiller
import models.models
import models.utils

logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(
            dataset_name, cfg, True, output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(
            dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(
                cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(
            model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info(
                "Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, teacher, resume=False):
    model.train()
    optimizer = my_build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    teacher_opt = my_build_optimizer(cfg.MODEL.DISTILLER, teacher)
    teacher_sche = build_lr_scheduler(cfg.MODEL.DISTILLER, teacher_opt)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler, teacher=teacher, teacher_opt=teacher_opt, teacher_sche=teacher_sche,
    )
    start_iter = (
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    if comm.get_world_size() > 1:
        module = model.module
        tea_module = teacher.module
    else:
        module = model
        tea_module = teacher

    if start_iter == 0:
        # load teacher during startup. optional features
        if cfg.MODEL.DISTILLER.PRELOAD_TEACHER != '':
            teacher_ckpt = torch.load(
                cfg.MODEL.DISTILLER.PRELOAD_TEACHER, map_location='cpu')
            assert cfg.MODEL.DISTILLER.PRELOAD_TYPE in ['all', 'teacher_only']
            if cfg.MODEL.DISTILLER.PRELOAD_TYPE == 'all':
                # load teacher and distiller
                mskeys, unexp_keys = teacher.load_state_dict(
                    teacher_ckpt['teacher'])
                if comm.is_main_process():
                    logger.info(
                        'Loading Pretrained Teacher Module Finished (all) !!')
                    logger.info('Loaded From: <%s> \n Loaded keys: <%s>' % (
                        cfg.MODEL.DISTILLER.PRELOAD_TEACHER, teacher_ckpt['teacher'].keys()))
                    logger.warning(
                        'Incompatible: <%s> \n Unexpected: <%s>' % (mskeys, unexp_keys))
            elif cfg.MODEL.DISTILLER.PRELOAD_TYPE == 'teacher_only':
                # only load teacher
                # NOTE: here we assume teacher is trained with DDP
                teacher_ckpt = {k.replace('module.teacher.', ''): v for k,
                                v in teacher_ckpt['teacher'].items() if 'module.teacher.' in k}
                mskeys, unexp_keys = tea_module.teacher.load_state_dict(
                    teacher_ckpt)
                if comm.is_main_process():
                    logger.info(
                        'Loading Pretrained Teacher Module Finished (teacher_only) !!')
                    logger.info('Loaded From: <%s> \n Loaded keys: <%s>' % (
                        cfg.MODEL.DISTILLER.PRELOAD_TEACHER, teacher_ckpt.keys()))
                    logger.warning(
                        'Incompatible: <%s> \n Unexpected: <%s>' % (mskeys, unexp_keys))

        if cfg.MODEL.DISTILLER.PRELOAD_FPN:
            # load teacher FPN to the student
            ckpt = {k: v for k, v in tea_module.teacher.fpn[0].state_dict(
            ).items() if 'bottom_up' not in k}
            origin_dict = module.backbone.state_dict()
            origin_dict.update(ckpt)
            mskeys, unexp_keys = module.backbone.load_state_dict(origin_dict)
            if comm.is_main_process():
                logger.info('Loading Pretrained FPN Module Finished !!')
                logger.warning(
                    'Incompatible Keys: <%s>' % [k for k in tea_module.teacher.fpn[0].state_dict() if k not in origin_dict])
                logger.warning(
                    'Incompatible: <%s> \n Unexpected: <%s>' % (mskeys, unexp_keys))

        if cfg.MODEL.DISTILLER.PRELOAD_HEAD:
            # load teacher detection head to the student
            origin_dict = module.state_dict()
            ckpt = {k: v for k, v in tea_module.teacher.pretrained_model[0].state_dict(
            ).items() if ('backbone' not in k) and ('pixel_' not in k) and (k in origin_dict)}
            origin_dict.update(ckpt)
            mskeys, unexp_keys = module.load_state_dict(origin_dict)

            if comm.is_main_process():
                logger.info('Loading Pretrained Detection Head Finished !!')
                logger.warning(
                    'Incompatible Keys: <%s>' % [k for k in tea_module.teacher.pretrained_model[0].state_dict() if k not in origin_dict])
                logger.warning(
                    'Incompatible: <%s> \n Unexpected: <%s>' % (mskeys, unexp_keys))

    pretrain_iter = cfg.MODEL.DISTILLER.PRETRAIN_TEACHER_ITERS
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter+pretrain_iter
    )

    writers = (
        [
            AllMetricPrinter(logger, max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    buffer_fpn = [None]

    def _f_fpn_hook(m, xin, xout):
        if m.train():
            buffer_fpn[0] = xout
    module.backbone.register_forward_hook(_f_fpn_hook)
    buffer_raw = [None]

    def _f_raw_hook(m, xin, xout):
        if m.train():
            buffer_raw[0] = xout
    module.backbone.bottom_up.register_forward_hook(_f_raw_hook)

    PIXEL_MEAN = tea_module.pixel_mean
    PIXEL_STD = tea_module.pixel_std

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(max(0, start_iter - pretrain_iter)) as storage:
        for data, _iteration in zip(data_loader, range(start_iter, max_iter + pretrain_iter)):
            if _iteration < pretrain_iter:
                teacher_opt.zero_grad()
                loss_dict = teacher(data, teacher_only=True)
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict
                losses.backward()

                if cfg.MODEL.DISTILLER.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model" and cfg.MODEL.DISTILLER.SOLVER.CLIP_GRADIENTS.CLIP_VALUE > 0.0:
                    norm = torch.nn.utils.clip_grad_norm_(
                        teacher.parameters(), cfg.MODEL.DISTILLER.SOLVER.CLIP_GRADIENTS.CLIP_VALUE)
                    storage.put_scalar('log_tea_grad_norm', norm)

                teacher_opt.step()

                loss_dict_reduced = {k: v.item()
                                     for k, v in comm.reduce_dict(loss_dict).items()}

                if _iteration % 20 == 0:
                    logger.info('Pretrain Teacher: <%d/%d> %s' %
                                (_iteration, pretrain_iter, loss_dict_reduced))

                periodic_checkpointer.step(_iteration)
                continue

            iteration = _iteration - pretrain_iter
            storage.iter = iteration

            fix_backbone = iteration < cfg.MODEL.DISTILLER.FIX_BACKBONE_BEFORE
            fix_head = iteration < cfg.MODEL.DISTILLER.FIX_HEAD_BEFORE
            forward_only = False

            if iteration > cfg.MODEL.DISTILLER.STOP_JOINT:
                forward_only = True

            loss_dict = model(data)

            images = [x['image'].to('cuda') for x in data]
            images = [(x - PIXEL_MEAN) / PIXEL_STD for x in images]
            SD = module.backbone.size_divisibility
            images = ImageList.from_tensors(
                images, SD)
            raw_feat = {
                'backbone_feat': buffer_raw[0],
                'fpn_feat': buffer_fpn[0],
                'batched_inputs': data,
                'images': images
            }
            raw_feat['iteration'] = iteration

            loss_tea_dict = teacher(raw_feat, forward_only=forward_only)
            del raw_feat

            loss_dict.update(loss_tea_dict)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            optimizer.zero_grad()
            teacher_opt.zero_grad()
            losses.backward()

            loss_dict_reduced = {k: v.item()
                                 for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)

            if fix_backbone:
                for p in module.backbone.parameters():
                    p.grad = None

            if fix_head:
                for p in module.head.parameters():
                    p.grad = None

            if forward_only:
                for p in tea_module.teacher.parameters():
                    p.grad = None

            if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model" and cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE > 0.0:
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE)
                storage.put_scalar('log_grad_norm', norm)

            if cfg.MODEL.DISTILLER.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model" and cfg.MODEL.DISTILLER.SOLVER.CLIP_GRADIENTS.CLIP_VALUE > 0.0:
                norm = torch.nn.utils.clip_grad_norm_(
                    teacher.parameters(), cfg.MODEL.DISTILLER.SOLVER.CLIP_GRADIENTS.CLIP_VALUE)
                storage.put_scalar('log_tea_grad_norm', norm)

            optimizer.step()
            teacher_opt.step()

            del loss_dict_reduced, losses_reduced, losses, loss_dict, loss_tea_dict
            buffer_fpn[0], buffer_raw[0] = None, None
            # torch.cuda.empty_cache()

            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            storage.put_scalar(
                "lr.tea", teacher_opt.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()
            if not forward_only:
                # NOTE: teacher is fixed, therefore no need to decay weight decay
                teacher_sche.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                comm.synchronize()

            if iteration + pretrain_iter - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(_iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg = build_distill_configs(cfg)

    cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)

    if len(cfg.EXTRA_CFG) > 0:
        for cf_file in cfg.EXTRA_CFG:
            cfg.merge_from_file(cf_file)

    # copy max iters
    cfg.MODEL.DISTILLER.SOLVER.MAX_ITER = cfg.SOLVER.MAX_ITER
    # if we use steps, we use the same
    cfg.MODEL.DISTILLER.SOLVER.STEPS = cfg.SOLVER.STEPS

    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    teacher = models.models.Distillator(cfg, model)

    model.to(torch.device(cfg.MODEL.DEVICE))
    teacher.to(torch.device(cfg.MODEL.DEVICE))

    logger.info("Model:\n{}".format(model))
    logger.info("Teacher:\n{}".format(teacher))

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
        )
        teacher = DistributedDataParallel(
            teacher, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
        )

    do_train(cfg, model, teacher, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    
    if args.machine_rank == 0:
        import subprocess
        master_ip = subprocess.check_output(['hostname', '--fqdn']).decode("utf-8")
        master_ip = str(master_ip).strip()
        args.dist_url = 'tcp://{}:23456'.format(master_ip)
        print(args.dist_url)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )