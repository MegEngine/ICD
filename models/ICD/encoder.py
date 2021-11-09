import megengine as mge
import megengine.module as M
from megengine import functional as F
import numpy as np
from abc import abstractmethod
from .layers import MLP
import math
import json


class InstanceHWEncoder(M.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden_dim = cfg.distiller.HIDDEN_DIM

        with open('./distill_configs/coco_obj.json') as f:
            coco_position_prior = json.load(f)

        self.coco_pos_prior = np.array([[coco_position_prior[x][0], coco_position_prior[x][1]]
                                        for x in sorted(list(coco_position_prior.keys()))])
        coco_class_prior = np.array([coco_position_prior[x][2]
                                     for x in sorted(list(coco_position_prior.keys()))])
        self.coco_class_prior = coco_class_prior / coco_class_prior.sum()


        self.neg_ratio = 5
        self.num_pos_feats = 128
        self.scale = 2 * math.pi
        self.temperature = 10000
        self.num_classes = cfg.distiller.NUM_CLASSES
        self.max_boxes = cfg.distiller.MAX_LABELS
        #self.add_scale_indicator = cfg.distiller.ADD_SCALE_INDICATOR

        #enc_type = cfg.distiller.ENCODER_TYPE
        #assert enc_type in ['mlp', 'encoder']

        inp_dim = self.num_classes + self.num_pos_feats * 2
        #if self.add_scale_indicator:
        inp_dim += 10

        self.encoder = MLP(inp_dim, hidden_dim, hidden_dim, 3)
        self.predictor = MLP(hidden_dim, hidden_dim, 3, 3)

    def get_pos_embeddings(self, x):
        # Input x : N, seq, 4
        # Output x: N, seq, self.pos_feat*4
        dim_t = F.arange(self.num_pos_feats, dtype='float32', device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = F.expand_dims(x, axis=-1) * self.scale / dim_t
        # actually dim_t is repeated over 2, so we can simple use sin and cos over each
        pos_x = F.concat([F.sin(pos_x[:, :, :, 0::2]), F.cos(pos_x[:, :, :, 1::2])], axis=3)
        pos_x = F.flatten(pos_x, start_axis=2)
        return pos_x

    def sample_box_like_coco(self, length, num_cls):
        samples_class = np.random.choice(num_cls, length, p=self.coco_class_prior)
        samples_pos_prob = self.coco_pos_prior[samples_class]  # K, 2, 4
        loc = samples_pos_prob[:, 0, :]
        std = samples_pos_prob[:, 1, :]
        samples_loc = np.random.normal(loc, std).reshape(-1, 4).clip(min=loc-2*std, max=loc+2*std)  # K, 4 [x, y, h, w]

        # Note that we use random x, y instead of statistic values
        samples_loc[:, :2] = np.random.rand(*samples_loc[:, :2].shape).clip(min=0.03, max=0.97)
        samples_loc[:, 2:] = samples_loc[:, 2:].clip(min=2e-3, max=0.7)
        # to x y x y
        final_loc = np.stack((samples_loc[:, 0]-samples_loc[:, 2]/2, samples_loc[:, 1]-samples_loc[:, 3]/2,
                              samples_loc[:, 0]+samples_loc[:, 2]/2, samples_loc[:, 1]+samples_loc[:, 3]/2)).T
        return samples_class, final_loc

    @abstractmethod
    def get_initial_descriptor(self, instances, image_size):
        pass

    @abstractmethod
    def loss(self, feat, ins_mask_gt, ins_mask, pos_gt):
        pass

    def forward(self, instances, pro_feats=None, image_size=None):
        # feat has shape (seq, bsz, C), mask (bsz, seq), gts (bsz, seq)
        feat, mask, gts, pos_gt = self.get_initial_descriptor(instances, image_size)
        feat = self.encoder(feat)
        return feat, mask, gts, pos_gt

class InstanceRegEncoder(InstanceHWEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        hidden_dim = cfg.distiller.HIDDEN_DIM
        self.predictor = MLP(hidden_dim, hidden_dim, 5, 3)

    def get_initial_descriptor(self, instances, image_size):
        box_descriptor = []
        class_descriptor = []
        batch_gts = []

        for _bs in range(instances.shape[0]):
            nr_boxes_cur_img = int(image_size[_bs, -1])
            H, W = image_size[_bs, 2], image_size[_bs, 3]
            # How many boxes after add fake descirptors
            nr_effective_boxes = min(
                max(1, int(nr_boxes_cur_img * (self.neg_ratio + 1))), self.max_boxes)

            sampled_cls, sampled_box = self.sample_box_like_coco(nr_effective_boxes, self.num_classes)
            bboxes = F.clip(mge.Tensor(sampled_box), lower=0, upper=1) * mge.tensor([W, H, W, H])

            r_bboxes = instances[_bs, :nr_boxes_cur_img, :4]
            device = r_bboxes.device
            bboxes[:nr_boxes_cur_img] = r_bboxes

            labels = mge.Tensor(sampled_cls, dtype='int32', device=device)
            r_labels = instances[_bs, :nr_boxes_cur_img, 4]  # start from 1 ? 

            labels[:nr_boxes_cur_img] = r_labels.astype("int32")

            # range from (0, 1)
            bboxes[:, [0, 2]] /= W
            bboxes[:, [1, 3]] /= H

            chw_bboxes = F.zeros_like(bboxes)
            # N x 6
            chw_bboxes = F.concat((chw_bboxes, chw_bboxes[:, :2]), 1)
            # to (cx, cy, hw, hy)
            chw_bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
            chw_bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2
            chw_bboxes[:, 2] = (bboxes[:, 2] - bboxes[:, 0])
            chw_bboxes[:, 3] = (bboxes[:, 3] - bboxes[:, 1])

            #NOTE(peizhen): fix scale range bug from low=0,high=1 to low=-0.3,high=0.3 via code review by kzj
            jitter = mge.random.uniform(low=-0.3, high=0.3, size=chw_bboxes[:, :2].shape)

            chw_bboxes[:, 4:] = chw_bboxes[:, :2] + jitter * chw_bboxes[:, 2:4]

            # (x1, y1, x2, y2, cx_, cy_)
            chw_bboxes[:, :4] = bboxes

            batch_gts.append(nr_boxes_cur_img)
            box_descriptor.append(chw_bboxes)
            class_descriptor.append(labels)

        max_seq_length = np.array([b.shape[0] for b in box_descriptor]).max()

        # 1 for True, 0 for Fake
        final_gts = F.zeros([len(instances), max_seq_length], device=device).detach() # detach
        # mask unhandled elements masks
        final_masks = F.ones([len(instances), max_seq_length], device=device, dtype='bool').detach() # detach
        final_cls_descriptors = F.zeros([len(instances), max_seq_length], device=device, dtype='int32')
        final_pos_descriptors = F.zeros([len(instances), max_seq_length, 2], device=device)
        final_hw_gt = F.zeros([len(instances), max_seq_length, 4], device=device).detach() # detach

        for i, (d_cls, d_box, n_gts) in enumerate(zip(class_descriptor, box_descriptor, batch_gts)):
            final_gts[i, :n_gts] = 1.0
            final_masks[i, :d_cls.shape[0]] = False
            final_cls_descriptors[i, :d_cls.shape[0]] = d_cls
            final_pos_descriptors[i, :d_cls.shape[0], :] = d_box[:, 4:]
            final_hw_gt[i, :d_cls.shape[0], :2] = d_box[:, 4:] - d_box[:, :2]
            final_hw_gt[i, :d_cls.shape[0], 2:4] = d_box[:, 2:4] - d_box[:, 4:]

        final_cls_descriptors = F.one_hot(final_cls_descriptors, self.num_classes)
        final_pos_descriptors = self.get_pos_embeddings(final_pos_descriptors)


        final_scale_descriptors = F.zeros([len(instances), max_seq_length, 10], device=device)
        for i, d_box in enumerate(box_descriptor):
            hw = d_box[:, 2:4] - d_box[:, :2]  # K, 2
            hw = d_box[:, 2:4] - d_box[:, :2]  # K, 2
            hw[:, 0] = hw[:, 0] * W
            hw[:, 1] = hw[:, 1] * H
            indicator = F.clip(((F.log(hw) * 1.44) - 5.0), 0.0, 4.0).astype('int32')
            # print(indicator)
            indicator = F.one_hot(
                indicator, 5).astype('float32').reshape(-1, 10)  # K, 2, 5
            final_scale_descriptors[i, :d_box.shape[0], :] = indicator
            

        final_pos_descriptors = F.concat((final_pos_descriptors, final_scale_descriptors), 2)

        final_descriptors = F.concat((final_cls_descriptors, final_pos_descriptors), 2).transpose(1, 0, 2) # detach

        # .detach() to resemble torch.no_grad()

        del final_cls_descriptors
        del final_pos_descriptors
        
        return final_descriptors.detach(), final_masks.detach(), final_gts.detach(), final_hw_gt.detach()


    def loss(self, feat, ins_mask_gt, ins_mask, pos_gt):
        # feat: [seq, bs, feat]
        f_pre = self.predictor(feat)
        # [seq, bs]
        confidence = f_pre[:, :, 0]

        # # [bsz, seq, feat]
        pred_pos = F.sigmoid(f_pre[:, :, 1:]).transpose(1, 0, 2)
        pos_gt_scale = pos_gt + pos_gt[:, :, [2, 3, 0, 1]]

        # pos_mask: [bs, seq] 1.0 for effective elements
        # ins_mask: [bs, seq] 1.0 for real elements
        positive_mask = (~ins_mask).astype('float32')

        # NOTE: we rescale the positive samples to control the hardness of our task

        loss_conf = (F.loss.binary_cross_entropy(confidence.T, ins_mask_gt, with_logits=True, reduction='none') * 
                        positive_mask).sum() / F.clip(positive_mask.sum(),lower=1e-6)

        # we normalzie the box relative to GT box, to handle large and samll objects
        # note that we use ins_mask_gt as offset masks, which ignores fake instances
        loss_reg = ((F.loss.l1_loss(pred_pos, pos_gt, reduction='none') \
                / F.clip(pos_gt_scale, lower=1e-7)).sum(-1) * ins_mask_gt).sum() / F.clip(ins_mask_gt.sum(), lower=1e-6)

        return {
            'aux.prop_conf': loss_conf,
            'aux.prop_reg': loss_reg
        }
