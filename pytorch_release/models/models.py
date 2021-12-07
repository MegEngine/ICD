from abc import abstractmethod
import torch
from torch import nn
from detectron2.utils.events import get_event_storage
from .utils import *
from .distiller import build_distiller
from .teacher import build_teacher

from detectron2.utils.events import EventWriter, get_event_storage


class Distillator(nn.Module):
    def __init__(self, cfg, student) -> None:
        super().__init__()
        self.cfg = cfg
        self.student_buffer = [student]  # as a printer

        self.teacher = build_teacher(cfg, student)

        distillers = []
        for dis_name in cfg.MODEL.DISTILLER.TYPES:
            distillers.append(build_distiller(
                cfg, dis_name, student, self.teacher))

        self.distillers = nn.ModuleList(distillers)

        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    def forward(self, raw_output, forward_only=False, teacher_only=False):
        '''
        Input:
            batched_inputs, images, r_features, features, gts
        Output:
            losses_tea      : loss dict
            r_features_tea  : features from backbone
            features_tea    : features from FPN
        '''
        if teacher_only:
            loss_dict, _ = self.teacher(raw_output, None, None, None)
            return loss_dict

        r_feats = raw_output['backbone_feat']
        fpn_feats = raw_output['fpn_feat']
        batched_inputs = raw_output['batched_inputs']
        images = raw_output['images']
        iteration = raw_output['iteration']

        if iteration < self.cfg.MODEL.DISTILLER.BYPASS_DISTILL or iteration > self.cfg.MODEL.DISTILLER.BYPASS_DISTILL_AFTER:
            distill_flag = self.cfg.MODEL.DISTILLER.DISTILL_OFF
        else:
            distill_flag = self.cfg.MODEL.DISTILLER.DISTILL_ON

        raw_output['distill_flag'] = distill_flag

        storage = get_event_storage()
        storage.put_scalar('distill_flag', distill_flag, False)

        if forward_only:
            with torch.no_grad():
                loss_dict, feat_dict_tea = self.teacher(
                    batched_inputs, images, r_feats, fpn_feats)
        else:
            loss_dict, feat_dict_tea = self.teacher(
                batched_inputs, images, r_feats, fpn_feats)

        for i, distiller in enumerate(self.distillers):
            loss_d = distiller(raw_output, feat_dict_tea)
            loss_d = {'distill.%s.%s' % (i, k): v for k, v in loss_d.items()}
            loss_dict.update(loss_d)

        return loss_dict
