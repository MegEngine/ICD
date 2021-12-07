# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN
from detectron2.utils.events import EventWriter, get_event_storage
import numpy as np
import torch.utils.data

import torch.nn as nn

from detectron2.utils.env import seed_all_rng

from detectron2.evaluation.evaluator import *
from detectron2.solver.build import maybe_add_gradient_clipping

"""
This file contains the default logic to build a dataloader for training or testing.
"""


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)


def my_build_optimizer(cfg, model):
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    if isinstance(model, nn.Module):
        model = [model]

    for m in model:
        for key, value in m.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR

            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            # if "backbone" in key or "encoder" in key:
            #     lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr,
                        "weight_decay": weight_decay}]

    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    elif optimizer_type == "ADAMW":
        optimizer = torch.optim.AdamW(params, cfg.SOLVER.BASE_LR)
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer


class AllMetricPrinter(EventWriter):
    """
    Print **common** metrics to the terminal, including
    iteration time, ETA, memory, all losses, and the learning rate.
    It also applies smoothing using a window of 20 elements.

    It's meant to print common metrics in common ways.
    To print something in more customized ways, please implement a similar printer by yourself.
    """

    def __init__(self, logger, max_iter):
        """
        Args:
            max_iter (int): the maximum number of iterations to train.
                Used to compute ETA.
        """
        self.logger = logger
        self._max_iter = max_iter
        self._last_write = None

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter
        if iteration == self._max_iter:
            # This hook only reports training progress (loss, ETA, etc) but not other data,
            # therefore do not write anything after training succeeds, even if this method
            # is called.
            return

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None

        eta_string = None
        try:
            iter_time = storage.history("time").global_avg()
            eta_seconds = storage.history("time").median(
                1000) * (self._max_iter - iteration - 1)
            storage.put_scalar("eta_seconds", eta_seconds,
                               smoothing_hint=False)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            iter_time = None
            # estimate eta on our own - more noisy
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (
                    iteration - self._last_write[0]
                )
                eta_seconds = estimate_iter_time * \
                    (self._max_iter - iteration - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iteration, time.perf_counter())

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(
            " {eta}iter: {iter} {log}  {time}{data_time}  {memory}".format(
                eta=f"eta: {eta_string}  " if eta_string else "",
                iter=iteration,
                log="  ".join(
                    [
                        "{}: {:.4g}".format(k, v.median(20))
                        for k, v in storage.histories().items()
                        if k not in ['time', 'data_time', 'eta_seconds']
                    ]
                ),
                time="time: {:.4f}  ".format(
                    iter_time) if iter_time is not None else "",
                data_time="data_time: {:.4f}  ".format(
                    data_time) if data_time is not None else "",
                memory="max_mem: {:.0f}M".format(
                    max_mem_mb) if max_mem_mb is not None else "",
            )
        )


def build_distill_configs(cfg):
    cfg.EXTRA_CFG = []
    # load extra configs for convenience

    cfg.MODEL.DISTILLER = CN()
    cfg.SOLVER.OPTIMIZER = 'SGD'

    cfg.MODEL.DISTILLER.PRETRAIN_TEACHER_ITERS = 0
    # pretrain teacher for extra iterations (with maximum learning rate)
    # remember to set teacher warmup to zero !
    cfg.MODEL.DISTILLER.BYPASS_DISTILL = 1000
    # when to start distillation
    cfg.MODEL.DISTILLER.BYPASS_DISTILL_AFTER = 99999999
    # when to disable distillation
    cfg.MODEL.DISTILLER.STOP_JOINT = 9999999
    # when to stop joint training (set teachers loss to 0)

    cfg.MODEL.DISTILLER.FIX_BACKBONE_BEFORE = 0
    cfg.MODEL.DISTILLER.FIX_HEAD_BEFORE = 0

    cfg.MODEL.DISTILLER.TEACHER = 'ModelTeacher_II'
    # transconv, trans, small, meanteacher ..
    cfg.MODEL.DISTILLER.TYPES = ['InstanceConditionalDistillation']

    cfg.MODEL.DISTILLER.PRELOAD_TEACHER = ''
    cfg.MODEL.DISTILLER.PRELOAD_TYPE = 'teacher_only'
    # 'all', 'teacher_only'

    # bag of tricks
    cfg.MODEL.DISTILLER.PRELOAD_FPN = True
    cfg.MODEL.DISTILLER.PRELOAD_HEAD = True

    cfg.MODEL.DISTILLER.IGNORE_DISTILL = ['']

    # model distiller
    cfg.MODEL.DISTILLER.MODEL_DISTILLER_CONFIG = 'COCO-Detection/retinanet_R_101_FPN_3x.yaml'
    cfg.MODEL.DISTILLER.MODEL_LOAD_OFFICIAL = True
    # load model from MODEL_ZOO or configs.weights

    _C = cfg.MODEL.DISTILLER
    build_tea_optimizer(_C)
    ############################ INSTANCE DISTILL ################################

    cfg.MODEL.DISTILLER.INS = CN()

    cfg.MODEL.DISTILLER.INS.TASK_NAME = 'InstanceEncoder'

    cfg.MODEL.DISTILLER.INS.DECODER = 'DecoderWrapper'
    cfg.MODEL.DISTILLER.INS.DECODER_POSEMB_ON_V = False
    cfg.MODEL.DISTILLER.INS.DISTILL_WITH_POS = False

    cfg.MODEL.DISTILLER.INS.ADD_SCALE_INDICATOR = True

    cfg.MODEL.DISTILLER.INS.DISTILL_NORM = 'ln'

    cfg.MODEL.DISTILLER.INS.NUM_CLASSES = 80
    cfg.MODEL.DISTILLER.INS.UNIFORM_SAMPLE_CLASS = False
    cfg.MODEL.DISTILLER.INS.UNIFORM_SAMPLE_BOX = False

    cfg.MODEL.DISTILLER.INS.INPUT_FEATS = ['p3', 'p4', 'p5', 'p6', 'p7']
    # NOTE: This is fundermental

    cfg.MODEL.DISTILLER.INS.ATT_LAYERS = 1
    # 0 for multi-head att, > 0 for cascade of N layers of transformer decoder

    cfg.MODEL.DISTILLER.INS.PROJECT_POS = True
    # project position embeddings

    cfg.MODEL.DISTILLER.INS.SAMPLE_RULE = 'relative'
    # negative sampling rule: {fixed, relative}
    cfg.MODEL.DISTILLER.INS.NUM_NEG_POS_RATIO = 5.0
    cfg.MODEL.DISTILLER.INS.NUM_LABELS = 100
    cfg.MODEL.DISTILLER.INS.MAX_LABELS = 300

    cfg.MODEL.DISTILLER.INS.DROPOUT = 0.0

    cfg.MODEL.DISTILLER.INS.CONFIDENCE_LOSS_WEIGHT = 1.0

    cfg.MODEL.DISTILLER.INS.USE_POS_EMBEDDING = True

    cfg.MODEL.DISTILLER.INS.ATT_HEADS = 8
    cfg.MODEL.DISTILLER.INS.HIDDEN_DIM = 256
    # following common practice

    cfg.MODEL.DISTILLER.INS.VALUE_RECONST = -1.0

    ######################## Loss ############################
    cfg.MODEL.DISTILLER.DISTILL_OFF = 0
    cfg.MODEL.DISTILLER.DISTILL_ON = 1
    # 0: no distill but update adapter head
    # 1: distill student
    # 2: distill teachers
    # others: two way backward

    cfg.MODEL.DISTILLER.INS_ATT_MIMIC = CN({
        'WEIGHT_VALUE': 8.0,

        'TEMP_MASK': 1.0,
        'TEMP_VALUE': 1.0,
        'TEMP_DECAY': False,
        'TEMP_DECAY_TO': 1.0,

        'DISTILL_NEGATIVE': False,

        'LOSS_FORM': 'mse',
        'SAMPLE_RANGE': 0.3,
    })

    return cfg


def build_tea_optimizer(_C):
    # This is copied from detectron2 and modified according to DETR

    _C.SOLVER = CN()
    # See detectron2/solver/build.py for LR scheduler options
    # Follows DETR settings
    _C.SOLVER.OPTIMIZER = 'ADAMW'
    # Step LR causes significant change in few iters, we use cosine for simplicity
    _C.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    # NOTE: MAX_ITER will to rewrite automatically
    _C.SOLVER.MAX_ITER = -1
    _C.SOLVER.BASE_LR = 1e-4
    _C.SOLVER.MOMENTUM = 0.9
    _C.SOLVER.NESTEROV = False

    _C.SOLVER.WEIGHT_DECAY = 0.0001
    # The weight decay that's applied to parameters of normalization layers
    # (typically the affine transformation)
    _C.SOLVER.WEIGHT_DECAY_NORM = 0.0

    _C.SOLVER.GAMMA = 0.1
    # The iteration number to decrease learning rate by GAMMA.
    _C.SOLVER.STEPS = (30000,)

    # All follows DETR settings
    _C.SOLVER.WARMUP_FACTOR = 1.0
    _C.SOLVER.WARMUP_ITERS = 10
    _C.SOLVER.WARMUP_METHOD = "linear"

    # Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
    # biases. This is not useful (at least for recent models). You should avoid
    # changing these and they exist only to reproduce Detectron v1 training if
    # desired.
    _C.SOLVER.BIAS_LR_FACTOR = 1.0
    _C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

    # Gradient clipping
    _C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": True})
    # Type of gradient clipping, currently 2 values are supported:
    # - "value": the absolute values of elements of each gradients are clipped
    # - "norm": the norm of the gradient for each parameter is clipped thus
    #   affecting all elements in the parameter
    _C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
    # Maximum absolute value used for clipping gradients
    _C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.01
    # Floating point number p for L-p norm to be used with the "norm"
    # gradient clipping type; for L-inf, please specify .inf
    _C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
