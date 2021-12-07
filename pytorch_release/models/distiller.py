import torch
from torch import nn
from torch.nn import functional as F

import torch.nn.functional as F
from .utils import *


from detectron2.utils.registry import Registry

DISTILLER_REGISTRY = Registry("DISTILLER")  # noqa F401 isort:skip
DISTILLER_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_distiller(cfg, name, student, teacher):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    model = DISTILLER_REGISTRY.get(name)(cfg, student, teacher)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model


@DISTILLER_REGISTRY.register()
class InstanceConditionalDistillation(nn.Module):
    """
    Distillation with multi-head attention. Mimic attention and features. 
    """

    def __init__(self, cfg, student, teacher) -> None:
        super().__init__()
        self.cfg = cfg
        self.student = [student]

        self.cfg = cfg
        hidden_dim = cfg.MODEL.DISTILLER.INS.HIDDEN_DIM

        self.pos_embedding = PositionEmbeddingSine(
            hidden_dim // 2, normalize=True)

        self.teacher_ptr = [teacher]
        self.attention_module = build_decoder_module(
            cfg)

        self.feat_keys = cfg.MODEL.DISTILLER.INS.INPUT_FEATS

        self.weight_value = cfg.MODEL.DISTILLER.INS_ATT_MIMIC.WEIGHT_VALUE
        self.temp_value = cfg.MODEL.DISTILLER.INS_ATT_MIMIC.TEMP_VALUE
        if self.temp_value < 0:
            self.temp_value = nn.Parameter(torch.ones([1]).mean())

        self.distill_norm_type = cfg.MODEL.DISTILLER.INS.DISTILL_NORM

        self.distill_negative = cfg.MODEL.DISTILLER.INS_ATT_MIMIC.DISTILL_NEGATIVE
        self.use_pos_embds = cfg.MODEL.DISTILLER.INS.USE_POS_EMBEDDING

        self.predictor = MLP(hidden_dim, hidden_dim, 1, 3)

        if self.distill_norm_type == 'ln':
            self.distill_norm_ = nn.LayerNorm(
                [hidden_dim // cfg.MODEL.DISTILLER.INS.ATT_HEADS], elementwise_affine=False)
            self.distill_norm_tea = nn.LayerNorm(
                [hidden_dim // cfg.MODEL.DISTILLER.INS.ATT_HEADS], elementwise_affine=False)
        elif self.distill_norm_type == 'tln':
            self.distill_norm_ = nn.Sequential()
            self.distill_norm_tea = nn.LayerNorm(
                [hidden_dim // cfg.MODEL.DISTILLER.INS.ATT_HEADS], elementwise_affine=False)
        else:
            self.distill_norm_ = nn.Sequential()
            self.distill_norm_tea = nn.Sequential()

        self.loss_form = cfg.MODEL.DISTILLER.INS_ATT_MIMIC.LOSS_FORM

    def concate_multiscale_reps(self, feat, pos_emb, mask):
        # permute and concate features form multiscale to a tensor under transformer definition
        keys = self.feat_keys

        feat = torch.cat([feat[k].flatten(2).permute(2, 0, 1)
                          for k in keys], 0)  # S, N, C
        pos_emb = torch.cat([pos_emb[k].flatten(2).permute(
            2, 0, 1) for k in keys], 0)  # S, N, C
        mask = torch.cat([mask[k].flatten(2).squeeze(1)
                          for k in keys], 1)  # N, S
        return feat, pos_emb, mask

    def bce_identification_loss(self, feat_list, ins_mask, ins_mask_gt):
        # this is the identification loss that identifies a given instance is real or fake
        positive_mask = (~ins_mask).float()

        loss_dict = {}
        for i, dfeat in enumerate(feat_list):
            f_pre = self.predictor(dfeat)

            loss = (F.binary_cross_entropy_with_logits(f_pre.squeeze(-1).T, ins_mask_gt, reduction='none') *
                    positive_mask).sum() / positive_mask.sum()

            loss_dict['stu_bce.%s.loss' % i] = loss

        return loss_dict

    def mimic_loss(self, svalue, tvalue, value_mask):
        # value: num_seq, bsz, heads, channel
        # mask: [bsz, heads, 1, Seq]
        #value_mask = value_mask ** self.power_factor
        if self.loss_form in ['mse', 'MSE']:
            return ((F.mse_loss(svalue, tvalue, reduction='none').permute(1, 2, 3, 0)
                     * value_mask).sum(-1) / value_mask.sum(-1).clamp(min=1e-6)).mean()
        elif self.loss_form in ['l1', 'L1']:
            return (F.l1_loss(svalue, tvalue, reduction='none').permute(1, 2, 3, 0)
                    * value_mask).mean(2).sum() / value_mask.sum().clamp(min=1e-6)
        elif self.loss_form in ['smoothL1']:
            return (F.smooth_l1_loss(svalue, tvalue, reduction='none').permute(1, 2, 3, 0)
                    * value_mask).mean(2).sum() / value_mask.sum().clamp(min=1e-6)
        elif self.loss_form in ['L2', 'l2']:
            return ((F.mse_loss(svalue, tvalue, reduction='none').permute(1, 2, 3, 0)
                     * value_mask).mean(2).sum() / value_mask.sum().clamp(min=1e-6)) ** 0.5

    def forward(self, features_dict, features_dict_tea):
        if isinstance(self.temp_value, nn.Parameter):
            self.temp_value.data = self.temp_value.data.clamp(min=0.1, max=8)
        else:
            if self.cfg.MODEL.DISTILLER.INS_ATT_MIMIC.TEMP_DECAY:
                decay_to = self.cfg.MODEL.DISTILLER.INS_ATT_MIMIC.TEMP_DECAY_TO
                ratio = features_dict['iteration'] / self.cfg.SOLVER.MAX_ITER
                self.temp_value = ratio * decay_to + \
                    (1 - ratio) * self.cfg.MODEL.DISTILLER.INS_ATT_MIMIC.TEMP_VALUE

        images = features_dict['images']
        batched_inputs = features_dict['batched_inputs']
        fpn_outputs = features_dict['fpn_feat']

        # assert set(self.feat_keys) == set(list(fpn_outputs.keys(
        # ))), 'WARNING: Unequal keys for fpn and attention ! <%s> != <%s>' % (self.feat_keys, fpn_outputs.keys())

        if features_dict['distill_flag'] == 0:
            fpn_outputs = {k: v.detach() for k, v in fpn_outputs.items()}

        # mask_out: zero for foreground, one for bg: BoolTensor(N, 1, H, W)
        mask_out = mask_out_padding(fpn_outputs, images)

        # fpn_outputs = self.scale_adapter(fpn_outputs)
        pos_embs = {k: self.pos_embedding(
            fpn_outputs[k], mask_out[k]) for k in self.feat_keys}
        # feat, pos: [S, N, C]; mask: [N, S]
        feat, pos_embs, mask_padding = self.concate_multiscale_reps(
            fpn_outputs, pos_embs, mask_out)

        # instance encoding: [K, N, C], ins_mask: bool[N, K], instance_gt: (0-1)[N, K]
        # NOTE: (0 for Fake Instance) in ins_mask
        ins_feat, ins_mask, ins_mask_gt = features_dict_tea['aux_feat']['encoded_ins']
        ins_feat = ins_feat.detach()

        if self.distill_negative:
            ins_mask_gt = (~ins_mask).detach().float()
            max_ele = None  # slice to the last element
        else:
            # calculate an element mask to reduce unnessessary computation
            max_ele = ins_mask_gt.long().sum(-1).max().item()

        # Note that mask is not normalized by softmax

        decoded_feat_list, att_mask_list, value_list = self.attention_module(
            ins_feat[:max_ele, :, :], feat, feat, query_mask=ins_mask[:, :max_ele], key_padding_mask=mask_padding, pos_embedding=pos_embs, proj_only=True)

        decoded_value_tea = features_dict_tea['aux_feat']['decoded_value']
        decoded_mask_tea = features_dict_tea['aux_feat']['decoded_mask']

        loss_value = torch.tensor([0.0], device=ins_mask_gt.device).mean()
        for i, (tmask, svalue, tvalue) in enumerate(zip(decoded_mask_tea, value_list, decoded_value_tea)):
            tmask = tmask.detach()  # bsz, heads, num_ins, num_seq

            # num_seq, bsz, heads, channel
            tvalue = self.distill_norm_tea(tvalue)
            tvalue = tvalue.detach()

            if self.weight_value > 0:
                with torch.no_grad():
                    value_mask = ((tmask / self.temp_value).softmax(-1) *
                                  ins_mask_gt.unsqueeze(1).unsqueeze(-1)).sum(2, keepdim=True)
                    # [bsz, heads, ins, Seq]

                svalue = self.distill_norm_(svalue)
                loss_value += self.mimic_loss(svalue,
                                              tvalue, value_mask) * self.weight_value

        loss_dict = {
            'matt.value': loss_value / len(decoded_feat_list),
        }

        if isinstance(self.temp_value, nn.Parameter):
            loss_dict['temp.value'] = self.temp_value.detach()

        return loss_dict
