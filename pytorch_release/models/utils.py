import numpy as np
import torch as torch
import torch.nn.functional as F
import torch.nn as nn
from detectron2.layers import Conv2d, ShapeSpec, get_norm
import math
from .layers.transformer import MultiheadAttention
import json
import torch.distributed as dist
from detectron2.utils.registry import Registry

INS_ENCODER_REGISTRY = Registry("INS_DECODER")  # noqa F401 isort:skip
INS_ENCODER_REGISTRY.__doc__ = ""

DECODER_MODULE_REGISTRY = Registry("DECODER_MODULE")  # noqa F401 isort:skip
DECODER_MODULE_REGISTRY.__doc__ = ""


def build_instance_encoder(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.DISTILLER.INS.TASK_NAME
    model = INS_ENCODER_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model


def build_decoder_module(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.DISTILLER.INS.DECODER
    model = DECODER_MODULE_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model


def mask_out_padding(fpn_features, images):
    # NOTE: zeros for forground
    image_sizes = images.image_sizes
    device = images.tensor.device
    h_, w_ = images.tensor.shape[-2:]
    masks = {}
    #assert len(feature_shapes) == len(self.feature_strides)
    for k, feat in fpn_features.items():
        # stride = 2 ** int(k[-1])
        N, _, H, W = feat.shape
        masks_per_feature_level = torch.ones(
            (N, H, W), dtype=torch.bool, device=device)
        stride = (h_/H + w_/W) / 2
        for img_idx, (h, w) in enumerate(image_sizes):
            masks_per_feature_level[
                img_idx,
                : int(np.ceil(float(h) / stride)),
                : int(np.ceil(float(w) / stride)),
            ] = 0
        masks[k] = masks_per_feature_level.unsqueeze(1)
    return masks


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    @torch.no_grad()
    def forward(self, x, mask):
        assert mask is not None
        not_mask = ~mask.squeeze(1)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class OfficialDecoderLayer(nn.Module):
    'Decoder with linear layer, that supports cascade, note that we ignore self-attentions, we assume each proposal corrospond to an independant prediction.'

    def __init__(self, cfg):
        super().__init__()
        channels = cfg.MODEL.DISTILLER.INS.HIDDEN_DIM
        heads = cfg.MODEL.DISTILLER.INS.ATT_HEADS
        layers = cfg.MODEL.DISTILLER.INS.ATT_LAYERS

        self.matt = MultiheadAttention(
            channels, heads, dropout=cfg.MODEL.DISTILLER.INS.DROPOUT)

        if cfg.MODEL.DISTILLER.INS.PROJECT_POS:
            self.pos_projector = nn.Linear(channels, channels)
        else:
            self.pos_projector = nn.Sequential()

        self.use_pos = cfg.MODEL.DISTILLER.INS.USE_POS_EMBEDDING

        self.linear_in = nn.Linear(channels, channels * 4)
        self.linear_out = nn.Linear(channels * 4, channels)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.pos_on_v = cfg.MODEL.DISTILLER.INS.DECODER_POSEMB_ON_V

    def with_pos_embed(self, tensor, pos):
        if not self.use_pos:
            return tensor
        pos = self.pos_projector(pos)
        return tensor if pos is None else tensor + pos

    def forward(self, q, k, v, key_padding_mask=None, pos_embedding=None, proj_only=False):
        k = self.with_pos_embed(k, pos_embedding)
        if self.pos_on_v:
            v = self.with_pos_embed(v, pos_embedding)
        q_, mask, values = self.matt(
            q, k, v, key_padding_mask=key_padding_mask, proj_only=proj_only)

        q = self.norm1(q + q_)
        q_ = self.linear_out(F.relu(self.linear_in(q)))
        q = self.norm2(q + q_)

        return q, mask, values


@DECODER_MODULE_REGISTRY.register()
class DecoderWrapper(nn.Module):
    # wrap decoder like multi-head attention
    def __init__(self, cfg):
        super().__init__()
        # channels=256, heads=8, layers=3
        channels = cfg.MODEL.DISTILLER.INS.HIDDEN_DIM
        heads = cfg.MODEL.DISTILLER.INS.ATT_HEADS
        layers = cfg.MODEL.DISTILLER.INS.ATT_LAYERS
        self.layers = layers
        if layers < 1:
            # this is a local module derived from official implementation, we modify the last modules
            self.matt = MultiheadAttention(
                channels, heads, dropout=cfg.MODEL.DISTILLER.INS.DROPOUT)
            if cfg.MODEL.DISTILLER.INS.PROJECT_POS:
                self.pos_projector = nn.Linear(channels, channels)
            else:
                self.pos_projector = nn.Sequential()
            self.use_pos = cfg.MODEL.DISTILLER.INS.USE_POS_EMBEDDING
        else:
            self.matt = nn.ModuleList([
                OfficialDecoderLayer(cfg) for _ in range(self.layers)
            ])

        self.pos_on_v = cfg.MODEL.DISTILLER.INS.DECODER_POSEMB_ON_V

    def with_pos_embed(self, tensor, pos):
        if not self.use_pos:
            return tensor
        pos = self.pos_projector(pos)
        return tensor if pos is None else tensor + pos

    def forward(self, q, k, v, query_mask=None, key_padding_mask=None, pos_embedding=None, proj_only=False):
        # q, v: [sequence_len, batch_size, channels]
        if self.layers < 1:
            k = self.with_pos_embed(k, pos_embedding)
            if self.pos_on_v:
                v = self.with_pos_embed(v, pos_embedding)
            att, mask, values = self.matt(
                q, k, v, key_padding_mask=key_padding_mask, proj_only=proj_only)
            return [att], [mask], [values]

        feats, masks, values = [], [], []
        for matt in self.matt:
            feat, mask, value = matt(
                q, k, v, key_padding_mask=key_padding_mask, pos_embedding=pos_embedding, proj_only=proj_only)
            q = feat
            feats.append(feat)
            masks.append(mask)
            values.append(value)

        return feats, masks, values


@INS_ENCODER_REGISTRY.register()
class InstanceEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.MODEL.DEVICE
        hidden_dim = cfg.MODEL.DISTILLER.INS.HIDDEN_DIM
        with open('./configs/coco_obj.json') as f:
            coco_position_prior = json.load(f)

        self.coco_pos_prior = np.array([[coco_position_prior[x][0], coco_position_prior[x][1]]
                                        for x in sorted(coco_position_prior.keys())])
        coco_class_prior = np.array([coco_position_prior[x][2]
                                     for x in sorted(coco_position_prior.keys())])
        self.coco_class_prior = coco_class_prior / coco_class_prior.sum()

        self.sample_rule = cfg.MODEL.DISTILLER.INS.SAMPLE_RULE
        self.neg_ratio = cfg.MODEL.DISTILLER.INS.NUM_NEG_POS_RATIO
        self.total_boxes = cfg.MODEL.DISTILLER.INS.NUM_LABELS
        self.max_boxes = cfg.MODEL.DISTILLER.INS.MAX_LABELS

        self.num_pos_feats = 128
        self.scale = 2 * math.pi
        self.temperature = 10000

        self.num_classes = cfg.MODEL.DISTILLER.INS.NUM_CLASSES

        if self.num_classes != 80 or cfg.MODEL.DISTILLER.INS.UNIFORM_SAMPLE_CLASS:
            self.coco_class_prior = np.ones(
                self.num_classes, dtype=np.float32) / self.num_classes
            self.coco_pos_prior = np.array([[coco_position_prior[str(x)][0], coco_position_prior[str(x)][1]]
                                            for x in sorted([int(y) for y in coco_position_prior.keys()])])

        if self.num_classes != 80 or cfg.MODEL.DISTILLER.INS.UNIFORM_SAMPLE_BOX:
            self.coco_pos_prior = np.ones(
                [self.num_classes, 2, 4], dtype=np.float32)
            self.coco_pos_prior[:, :, :] = 0.2
            # H, W: mean 0.5, std 0.2

        self.add_scale_indicator = cfg.MODEL.DISTILLER.INS.ADD_SCALE_INDICATOR

        inp_dim = self.num_classes + 128*2
        if self.add_scale_indicator:
            inp_dim += 10

        self.encoder = MLP(inp_dim, hidden_dim, hidden_dim, 3)

        # buffers, store gt labels
        self._tmp_gt_boxes = None
        self.predictor = MLP(hidden_dim, hidden_dim, 5, 3)

        self.conf_weight = cfg.MODEL.DISTILLER.INS.CONFIDENCE_LOSS_WEIGHT
        self.sample_range = cfg.MODEL.DISTILLER.INS_ATT_MIMIC.SAMPLE_RANGE

    def get_pos_embeddings(self, x):
        # Input x : N, seq, 4
        # Output x: N, seq, self.pos_feat*4
        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x[:, :, :, None] * self.scale / dim_t

        # actually dim_t is repeated over 2, so we can simple use sin and cos over each
        pos_x = torch.cat(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=3).flatten(2)

        return pos_x

    def sample_box_like_coco(self, len, num_cls):
        samples_class = np.random.choice(num_cls, len, p=self.coco_class_prior)
        samples_pos_prob = self.coco_pos_prior[samples_class]  # K, 2, 4
        loc = samples_pos_prob[:, 0, :]
        std = samples_pos_prob[:, 1, :]
        samples_loc = np.random.normal(
            loc, std).reshape(-1, 4).clip(min=loc-2*std, max=loc+2*std)  # K, 4 [x, y, h, w]

        # Note that we use random x, y instead of statistic values
        samples_loc[:, :2] = np.random.rand(
            *samples_loc[:, :2].shape).clip(min=0.03, max=0.97)
        samples_loc[:, 2:] = samples_loc[:, 2:].clip(min=2e-3, max=0.7)
        # to x y x y
        final_loc = np.stack((samples_loc[:, 0]-samples_loc[:, 2]/2, samples_loc[:, 1]-samples_loc[:, 3]/2,
                              samples_loc[:, 0]+samples_loc[:, 2]/2, samples_loc[:, 1]+samples_loc[:, 3]/2)).T
        return samples_class, final_loc

    @torch.no_grad()
    def get_initial_descriptor(self, batched_inputs):
        instances = [x['instances'] for x in batched_inputs]

        box_descriptor = []
        class_descriptor = []
        batch_gts = []

        for instance in instances:
            nr_boxes_cur_img = len(instance)

            H, W = instance.image_size

            # How many boxes after add fake descirptors
            if self.sample_rule == 'relative':
                nr_effective_boxes = min(
                    max(1, int(nr_boxes_cur_img * (self.neg_ratio + 1))), max(self.max_boxes, nr_boxes_cur_img))
            elif self.sample_rule == 'fixed':
                nr_effective_boxes = max(self.total_boxes, nr_boxes_cur_img)
            else:
                raise Exception(
                    'Not implemented sample rule: <%s>!' % self.sample_rule)

            sampled_cls, sampled_box = self.sample_box_like_coco(
                nr_effective_boxes, self.num_classes)
            bboxes = torch.tensor(sampled_box).clamp(
                min=0, max=1) * torch.tensor([W, H, W, H])
            bboxes = bboxes.to(self.device)

            r_bboxes = instance.gt_boxes.tensor.reshape(
                nr_boxes_cur_img, 4).to(self.device)
            bboxes[:nr_boxes_cur_img, :] = r_bboxes

            labels = torch.tensor(
                sampled_cls, dtype=torch.long).to(self.device)
            r_labels = instance.gt_classes.to(self.device)  # start from 1
            labels[:nr_boxes_cur_img] = r_labels

            # range from (0, 1)
            bboxes[:, [0, 2]] /= W
            bboxes[:, [1, 3]] /= H

            chw_bboxes = torch.zeros_like(bboxes)
            # N x 6
            chw_bboxes = torch.cat((chw_bboxes, chw_bboxes[:, :2]), 1)
            # to (cx, cy, hw, hy)
            chw_bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
            chw_bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2
            chw_bboxes[:, 2] = (bboxes[:, 2] - bboxes[:, 0])
            chw_bboxes[:, 3] = (bboxes[:, 3] - bboxes[:, 1])

            jitter = (torch.rand_like(
                chw_bboxes[:, :2]) - 0.5) * self.sample_range * 2.0
            chw_bboxes[:, 4:] = chw_bboxes[:, :2] + jitter * chw_bboxes[:, 2:4]

            # (x1, y1, x2, y2, cx_, cy_)
            chw_bboxes[:, :4] = bboxes

            batch_gts.append(nr_boxes_cur_img)
            box_descriptor.append(chw_bboxes)
            class_descriptor.append(labels)

        max_seq_length = np.array([b.size(0) for b in box_descriptor]).max()

        # 1 for True, 2 for Fake
        final_gts = torch.zeros(
            [len(instances), max_seq_length], device=self.device)
        # mask unhandled elements masks
        final_masks = torch.ones(
            [len(instances), max_seq_length], device=self.device, dtype=torch.bool)
        final_cls_descriptors = torch.zeros(
            [len(instances), max_seq_length], device=self.device, dtype=torch.long)
        final_pos_descriptors = torch.zeros(
            [len(instances), max_seq_length, 2], device=self.device)
        final_hw_gt = torch.zeros(
            [len(instances), max_seq_length, 4], device=self.device)

        for i, (d_cls, d_box, n_gts) in enumerate(zip(class_descriptor, box_descriptor, batch_gts)):
            final_gts[i, :n_gts] = 1.0
            final_masks[i, :d_cls.size(0)] = False
            final_cls_descriptors[i, :d_cls.size(0)] = d_cls
            final_pos_descriptors[i, :d_cls.size(0), :] = d_box[:, 4:]
            final_hw_gt[i, :d_cls.size(0), :2] = d_box[:, 4:] - d_box[:, :2]
            final_hw_gt[i, :d_cls.size(0), 2:4] = d_box[:, 2:4] - d_box[:, 4:]

        self.final_pos = final_pos_descriptors
        self._tmp_gt_cls = final_cls_descriptors.detach()

        final_cls_descriptors = F.one_hot(
            final_cls_descriptors, self.num_classes)
        final_pos_descriptors = self.get_pos_embeddings(final_pos_descriptors)

        if self.add_scale_indicator:
            # offset = torch.tensor(
            #     [0.04, 0.08, 0.16, 0.32, 0.64], device=self.device)
            final_scale_descriptors = torch.zeros(
                [len(instances), max_seq_length, 10], device=self.device)
            for i, d_box in enumerate(box_descriptor):
                hw = d_box[:, 2:4] - d_box[:, :2]  # K, 2
                hw[:, 0] = hw[:, 0] * W
                hw[:, 1] = hw[:, 1] * H
                indicator = ((hw.log() * 1.44) -
                             5.0).clamp(min=0.0, max=4.0).long()
                # print(indicator)
                indicator = F.one_hot(
                    indicator, 5).float().reshape(-1, 10)  # K, 2, 5
                final_scale_descriptors[i, :d_box.size(0), :] = indicator

            final_pos_descriptors = torch.cat(
                (final_pos_descriptors, final_scale_descriptors), 2)

        final_descriptors = torch.cat(
            (final_cls_descriptors, final_pos_descriptors), 2).permute(1, 0, 2)

        self._tmp_gt_boxes = final_hw_gt

        return final_descriptors, final_masks, final_gts

    def loss(self, feat, ins_mask_gt, ins_mask):
        assert self._tmp_gt_boxes is not None

        # feat: [seq, bs, feat]
        f_pre = self.predictor(feat)
        confidence = f_pre[:, :, 0]
        # # [bsz, seq, feat]
        pred_pos = f_pre[:, :, 1:].sigmoid().permute(1, 0, 2)

        pos_gt = self._tmp_gt_boxes
        pos_gt_scale = pos_gt + pos_gt[:, :, [2, 3, 0, 1]]

        # pos_mask: [bs, seq] 1.0 for effective elements
        # ins_mask: [bs, seq] 1.0 for real elements
        positive_mask = (~ins_mask).float()

        # NOTE: we rescale the positive samples to control the hardness of our task
        loss_conf = self.conf_weight * (F.binary_cross_entropy_with_logits(confidence.T, ins_mask_gt, reduction='none') *
                                        positive_mask).sum() / positive_mask.sum().clamp(min=1e-6)

        # we normalzie the box relative to GT box, to handle large and samll objects
        # note that we use ins_mask_gt as offset masks, which ignores fake instances
        loss_reg = ((F.l1_loss(pred_pos, pos_gt, reduction='none') / pos_gt_scale.clamp(min=1e-7)).sum(-1) *
                    ins_mask_gt).sum() / ins_mask_gt.sum().clamp(min=1e-6)

        # self._tmp_gt_boxes = None
        return {
            'prop_conf': loss_conf,
            'prop_reg': loss_reg
        }

    def forward(self, batched_inputs, pro_feats=None):
        # feat has shape (seq, bsz, C), mask (bsz, seq), gts (bsz, seq)
        feat, mask, gts = self.get_initial_descriptor(batched_inputs)

        self.final_pos = None

        feat = self.encoder(feat)
        return feat, mask, gts
