import os
from detectron2.config.config import CfgNode
from adet.config import get_cfg
import torch
from torch import nn
from torch.nn import functional as F
import detectron2.model_zoo

import torch.nn.functional as F
from .utils import *
from utils.build import build_distill_configs

from detectron2.utils.registry import Registry
from detectron2.structures import ImageList
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model


TEACHER_REGISTRY = Registry("TEACHER")  # noqa F401 isort:skip
TEACHER_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_teacher(cfg, parent):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.DISTILLER.TEACHER
    model = TEACHER_REGISTRY.get(meta_arch)(cfg, parent)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model


def detach_wrapper(x, module):
    set_requires_grad(module, False)
    y = module(x)
    set_requires_grad(module, True)
    return y


def set_requires_grad(m, flag):
    for p in m.parameters():
        p.requires_grad = flag


def get_tea_cfg(cfg):
    cfg_ = get_cfg()
    cfg_.MODEL.DISTILLER = CfgNode()
    cfg_ = build_distill_configs(cfg_)
    cfg_.merge_from_file(os.path.join(
        'configs', cfg.MODEL.DISTILLER.MODEL_DISTILLER_CONFIG), allow_unsafe=True)
    return cfg_


@TEACHER_REGISTRY.register()
class ModelTeacher(nn.Module):
    def __init__(self, cfg, parent=None):
        super().__init__()
        self.cfg = cfg

        if cfg.MODEL.DISTILLER.MODEL_LOAD_OFFICIAL:
            cfg_ = detectron2.model_zoo.get_config(
                cfg.MODEL.DISTILLER.MODEL_DISTILLER_CONFIG, trained=True)

            if cfg.MODEL.DISTILLER.MODEL_DISTILLER_CONFIG.endswith('.py'):
                cfg_.model.backbone.bottom_up.stem.norm = \
                    cfg_.model.backbone.bottom_up.stages.norm = \
                    cfg_.model.backbone.norm = "FrozenBN"

                cfg_.model.roi_heads.box_head.conv_norm = \
                    cfg_.model.roi_heads.mask_head.conv_norm = "FrozenBN"

            device = cfg.MODEL.DEVICE
            if device is not None and isinstance(cfg_, CfgNode):
                cfg_.MODEL.DEVICE = device

            # print(device)

            if isinstance(cfg_, CfgNode):
                model = build_model(cfg_)
                DetectionCheckpointer(model).load(cfg_.MODEL.WEIGHTS)
            else:
                from detectron2.config import instantiate
                model = instantiate(cfg_.model)
                if device is not None:
                    model = model.to(device)
                if "train" in cfg_ and "init_checkpoint" in cfg_.train:
                    DetectionCheckpointer(model).load(
                        cfg_.train.init_checkpoint)

            pretrained_model = model
        else:
            cfg_ = get_tea_cfg(cfg)
            cfg_.MODEL.DEVICE = device = cfg.MODEL.DEVICE

            pretrained_model = build_model(cfg_)
            DetectionCheckpointer(pretrained_model).load(cfg_.MODEL.WEIGHTS)

        # we only leave backbone for distillation
        # we do not record the parameters
        for p in pretrained_model.parameters():
            p.requires_grad = False

        self.pretrained_model = [pretrained_model]
        self.model = [pretrained_model.backbone.bottom_up]
        # NOTE: pretrained model do not have backbone !
        pretrained_model.backbone.bottom_up = nn.Sequential()
        self.fpn = [pretrained_model.backbone]

        # pretrained_model.pixel_mean #
        self.pixel_mean = torch.tensor(
            cfg_.MODEL.PIXEL_MEAN, device=self.cfg.MODEL.DEVICE).view(-1, 1, 1)
        self.pixel_mean.requires_grad = False
        # pretrained_model.pixel_std #torch.tensor(cfg_.MODEL.PIXEL_STD, device=self.cfg.MODEL.DEVICE).view(-1, 1, 1)
        self.pixel_std = torch.tensor(
            cfg_.MODEL.PIXEL_STD, device=self.cfg.MODEL.DEVICE).view(-1, 1, 1)
        self.pixel_std.requires_grad = False

    def forward(self, batched_inputs, images, raw_outputs, fpn_outputs):
        # NOTE: Maybe add support for JIT
        with torch.no_grad():
            images = [x["image"].to(self.cfg.MODEL.DEVICE)
                      for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]

            images = ImageList.from_tensors(
                images, self.fpn[0].size_divisibility)

            r_feat = self.model[0](images.tensor)
        with torch.no_grad():
            feat = self.fpn[0](r_feat)

        return {}, {'fpn_feat': feat, 'backbone_feat': r_feat, 'images': images}


@TEACHER_REGISTRY.register()
class ModelTeacher_II(ModelTeacher):
    """
    Instance Identification with model teacher
    use the pretrained model as teacher, train a feature extractor by identification task.
    """

    def __init__(self, cfg, parent=None):
        super().__init__(cfg, parent)
        hidden_dim = cfg.MODEL.DISTILLER.INS.HIDDEN_DIM
        self.parent_buffer = [parent]
        self.pos_embedding = PositionEmbeddingSine(
            hidden_dim//2, normalize=True)
        self.attention_module = build_decoder_module(
            cfg)  # DecoderWrapper(cfg)
        self.feat_keys = cfg.MODEL.DISTILLER.INS.INPUT_FEATS

        self.ins_encoder = build_instance_encoder(cfg)

        self.reconst_w = cfg.MODEL.DISTILLER.INS.VALUE_RECONST
        if self.reconst_w > 0:
            self.reconst_projector = nn.ModuleList([
                nn.Linear(hidden_dim, 256) for i in range(max(cfg.MODEL.DISTILLER.INS.ATT_LAYERS, 1))
            ])

    def concate_multiscale_reps(self, feat, pos_emb, mask):
        # permute and concate features form multiscale to a tensor in transformer definition
        keys = self.feat_keys

        feat = torch.cat([feat[k].flatten(2).permute(2, 0, 1)
                          for k in keys], 0)  # S, N, C
        pos_emb = torch.cat([pos_emb[k].flatten(2).permute(
            2, 0, 1) for k in keys], 0)  # S, N, C
        mask = torch.cat([mask[k].flatten(2).squeeze(1)
                          for k in keys], 1)  # N, S
        return feat, pos_emb, mask

    def forward(self, batched_inputs, images, raw_outputs, fpn_outputs):
        # get raw features
        _, tea_feat_dict = super().forward(
            batched_inputs, images, raw_outputs, fpn_outputs)

        images = tea_feat_dict['images']
        tea_raw_feat, tea_fpn_feat = tea_feat_dict['backbone_feat'], tea_feat_dict['fpn_feat']

        # mask_out: zero for foreground, one for bg: BoolTensor(N, 1, H, W)
        mask_out = mask_out_padding(tea_fpn_feat, images)

        pos_embs = {k: self.pos_embedding(
            tea_fpn_feat[k], mask_out[k]) for k in self.feat_keys}

        # feat, pos: [S, N, C]; mask: [N, S], Note that mask has not normalized by softmax
        feat_k, pos_embs, mask_padding = self.concate_multiscale_reps(
            tea_fpn_feat, pos_embs, mask_out)

        feat_v = feat_k

        # instance encoding: [K, N, C], ins_mask: bool[K, N], instance_gt: (0-1)[K, N]
        # NOTE: (0 for Fake Instance) in ins_mask
        ins_feat, ins_mask, ins_mask_gt = self.ins_encoder(
            batched_inputs, pro_feats=tea_fpn_feat)

        decoded_feat_list, att_mask_list, value_list = self.attention_module(
            ins_feat, feat_k, feat_v, query_mask=ins_mask, key_padding_mask=mask_padding, pos_embedding=pos_embs)

        loss_dict = self.loss(decoded_feat_list, ins_mask, ins_mask_gt)

        aux_feat = {
            'mask_out': mask_out,
            'pos_embs': pos_embs,
            'mask_padding': mask_padding,
            'encoded_ins': (ins_feat, ins_mask, ins_mask_gt),
            'decoded_feat': decoded_feat_list,
            'decoded_mask': att_mask_list,
            'decoded_value': value_list,
        }

        if self.reconst_w > 0:
            loss_dict['loss_reconst'] = self.loss_reconst(feat_v, value_list)

        return loss_dict, {'fpn_feat': tea_fpn_feat, 'backbone_feat': tea_raw_feat, 'aux_feat': aux_feat}

    def loss_reconst(self, feat_v, value_list):
        # This is an option motivated by Information Bottleneck, which minimizes reconstruction loss
        # feat_v : [seq_len, bsz, hidden_dim]
        feat_v = feat_v.detach()
        loss = 0.0
        for i, value in enumerate(value_list):
            # value : [seq_len, bsz, num_heads, head_dim]
            value = value.flatten(2)
            value = self.reconst_projector[i](value)
            loss += F.mse_loss(value, feat_v)

        return loss / (i + 1)

    def loss(self, feat_list, ins_mask, ins_mask_gt):
        # this is the identification loss that identifies a given instance is real or fake

        loss_dict = {}
        for i, dfeat in enumerate(feat_list):
            loss = self.ins_encoder.loss(dfeat, ins_mask_gt, ins_mask)
            loss = {'tea.%s.%s' % (i, k): v for k, v in loss.items()}
            loss_dict.update(loss)

        return loss_dict
