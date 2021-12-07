import megengine
import megengine as mge
from typing import Dict, List, Tuple
import megengine.module as M
import megengine.functional as F
import numpy as np
from .encoder import InstanceRegEncoder
from .decoder import DecoderWrapper
from .utility import PositionEmbeddingSine 


def mask_out_padding(fpn_features, images_sizes, images):
    # Modified from DETR: https://github.com/facebookresearch/detr/blob/main/LICENSE
    # NOTE: zeros for forground
    image_sizes = [(images_sizes[i, 2], images_sizes[i, 3]) for i in range(images_sizes.shape[0])]
    device = images_sizes.device
    h_, w_ = images.shape[-2:]
    masks = {}
    #assert len(feature_shapes) == len(self.feature_strides)
    for k, feat in fpn_features.items():
        # stride = 2 ** int(k[-1])
        N, _, H, W = feat.shape
        masks_per_feature_level = F.ones(
            (N, H, W), dtype='bool', device=device)
        stride = (h_ / H + w_ / W) / 2
        for img_idx, (h, w) in enumerate(image_sizes):
            masks_per_feature_level[
                img_idx,
                : int(np.ceil(float(h) / stride)),
                : int(np.ceil(float(w) / stride)),
            ] = 0
        masks[k] = F.expand_dims(masks_per_feature_level, 1) #masks_per_feature_level.unsqueeze(1)
    return masks


class ICD(M.Module):
    def __init__(self, hidden_dim, cfg):
        super().__init__()
        self.pos_embedding = PositionEmbeddingSine(
            num_pos_feats=hidden_dim // 2,
            normalize=True)

        self.ins_encoder = InstanceRegEncoder(cfg)
        self.attention_module_aux = DecoderWrapper(cfg)
        self.attention_module_distill = DecoderWrapper(cfg)
        # NOTE(peizhen): 1e-05 is not large enough and emprically might cause sqrt(neg) nan
        self.distill_norm_ = M.LayerNorm(
            [hidden_dim // cfg.distiller.ATT_HEADS], eps=1e-04, affine=False)
        #self.distill_norm_ = LayerNorm([hidden_dim // cfg.distiller.ATT_HEADS])
        self.feat_keys = cfg.distiller.FEAT_KEYS
        self.weight_value = cfg.distiller.WEIGHT_VALUE
        self.temp_value = cfg.distiller.TEMP_VALUE

        self.loss_keys = []
        self.num_losses = 3

    def mimic_loss(self, svalue, tvalue, value_mask):
        return (F.loss.square_loss(svalue, tvalue, reduction='none').transpose(1, 2, 3, 0)
                * value_mask).mean(2).sum() / F.clip(value_mask.sum(), lower=1e-6)

    def forward(self, features_dict_tea, features_dict_stu, images, instances, image_info, distill_flag=0):
        '''
        contain_box_mask: 1d float tensor, [1., 0., ...], denoting whether each image contain exactly objects
        nr_actual_boxes_per_img: list of int, exact object number each image contains
        '''
        nr_actual_boxes_per_img = [image_info[i, -1] for i in range(image_info.shape[0])]

        masks = mask_out_padding(features_dict_tea, image_info, images)

        pos_embs = {k: self.pos_embedding(
            features_dict_tea[k], masks[k]) for k in self.feat_keys}
        pos_emb = F.concat([F.transpose(F.flatten(pos_embs[k], 2), (2, 0, 1)) for k in self.feat_keys], 0).detach()  # S, N, C
        masks = F.concat([F.squeeze(F.flatten(masks[k], 2), 1)
                          for k in self.feat_keys], 1).detach()  # N, S

        loss_aux_dict, aux_info_dict = self.forward_aux(
            instances, features_dict_tea, image_info, {'mask_out': masks, 'pos_emb': pos_emb})
        loss_distill_dict = self.forward_distill(
            features_dict_stu, aux_info_dict, nr_actual_boxes_per_img, distill_flag, {'mask_out': masks, 'pos_emb': pos_emb})
        loss_aux_dict.update(loss_distill_dict)
        self.loss_keys = list(loss_aux_dict.keys())
        # print(self.loss_keys)
        return loss_aux_dict

    def forward_aux(self, instances, features_dict_tea, image_size, aux_input):
        # [S, N, C]
        feat = F.concat([F.flatten(features_dict_tea[k], start_axis=2).transpose(2, 0, 1)
                        for k in self.feat_keys], 0).detach()

        # instance encoding: [K, N, C], ins_mask: bool[N, K], instance_gt: (0-1)[N, K]
        # (0 for Fake Instance) in ins_mask

        # Below four variables provided by encoder forward have been detached before passing to here
        ins_feat, ins_mask, ins_mask_gt, pos_gt = self.ins_encoder(
            instances, pro_feats=features_dict_tea, image_size=image_size)
        decoded_feat, tmask, tvalue = self.attention_module_aux(
            ins_feat,
            feat,
            feat,
            query_mask=ins_mask,
            key_padding_mask=aux_input['mask_out'],
            pos_embedding=aux_input['pos_emb'])

        aux_info_dict = {
            'encoded_ins': (ins_feat, ins_mask, ins_mask_gt),
            'tmask': tmask,
            'tvalue': tvalue,
        }

        loss_dict = dict()
        loss_dict = self.ins_encoder.loss(
            decoded_feat, ins_mask_gt, ins_mask, pos_gt)

        return loss_dict, aux_info_dict


    def forward_distill(self, features_dict_stu, aux_info_dict, nr_actual_boxes_per_img, distill_flag, aux_input):
        loss_dict = dict()

        assert set(self.feat_keys) == set(list(features_dict_stu.keys(
        ))), 'WARNING: Unequal keys for fpn and attention ! <%s> != <%s>' % (self.feat_keys, features_dict_stu.keys())
        # [S, N, C]
        feat = F.concat([F.flatten(features_dict_stu[k], start_axis=2).transpose(2, 0, 1)
                        for k in self.feat_keys], 0)

        if distill_flag == 0:
            feat = feat.detach()

        # instance encoding: [K, N, C], ins_mask: bool[N, K], instance_gt: (0-1)[N, K]
        # (0 for Fake Instance) in ins_mask
        ins_feat, ins_mask, ins_mask_gt = aux_info_dict['encoded_ins']
        max_ele = int(max(max(nr_actual_boxes_per_img), 1))

        # Note that mask is not normalized by softmax
        # load state dict, therefore we share almost all parameters
        _, _, svalue = self.attention_module_distill(
            ins_feat[:max_ele, :, :],
            feat,
            feat,
            query_mask=ins_mask[:, :max_ele],
            key_padding_mask=aux_input['mask_out'],
            pos_embedding=aux_input['pos_emb'],
            proj_only=True)
        tvalue = aux_info_dict['tvalue']
        tmask = aux_info_dict['tmask']

        # [bsz, heads, ins, Seq]
        svalue = self.distill_norm_(svalue)
        # [Seq, bsz, heads, channel]
        tvalue = self.distill_norm_(tvalue)

        # cosine similarity between features, unreal instances are masked
        # feat are compact features for each instaces
        # value is weighted attention maps refactored as different heads
        # mask is q, k relation masks for distillation

        # [bsz, heads, 1, S]
        value_mask = (F.softmax(tmask / self.temp_value, axis=-1)
                      * F.expand_dims(F.expand_dims(ins_mask_gt, axis=1), axis=-1)
                      ).sum(2, keepdims=True).detach()
        # NOTE(peizhen): value_mask[j, ...] for any j-th image if it contains no image, beforehand, we could use a pseudo box for images who haven't any box
        # this should similarly apply to ins_encoder's loss auxiliary task loss too (no box then corresponding image should not contribute loss)

        # [bsz, heads, 1, num_seq]
        # value_mask = value_mask * contain_box_mask.reshape(-1, 1, 1, 1)
        loss_dict = {'distill': self.mimic_loss(
            svalue, tvalue.detach(), value_mask) * self.weight_value}
        return loss_dict
