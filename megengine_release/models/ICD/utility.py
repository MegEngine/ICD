import megengine as mge
import megengine.module as M
from megengine import functional as F
import numpy as np
import math
# mge.core.set_option('async_level', 0)


def safe_masked_fill(tensor: mge.Tensor, mask: mge.Tensor, val: float) -> mge.Tensor:
    '''
    same behavior as torch.tensor.masked_fill_(mask, val)
    '''
    assert mask.dtype == np.bool_
    discard_mask = ~mask
    keep_mask = mask.astype('float32')
    # NOTE(peizhen): simply tensor * ~mask + value * mask could not handle the value=float('+inf'/'-inf') case, since inf*0 = nan
    new_tensor = tensor * ~mask + F.where(mask, F.ones_like(mask) * val, F.zeros_like(mask))
    return new_tensor


def has_nan_or_inf(inp):
    invalid_mask = F.logical_or(F.isnan(inp), F.isinf(inp))
    return invalid_mask.sum().item() > 0


class PositionEmbeddingSine(M.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    Modified from DETR: https://github.com/facebookresearch/detr/blob/main/LICENSE
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

    def forward(self, x, mask):
        assert mask is not None
        not_mask = F.squeeze(~mask, 1)  # ~mask.squeeze(1)
        # import ipdb; ipdb.set_trace()
        y_embed = F.cumsum(not_mask.astype('int32'), 1)  # .cumsum(1, dtype=torch.float32)
        x_embed = F.cumsum(not_mask.astype('int32'), 2)  # .cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = F.arange(self.num_pos_feats,
                         dtype="float32", device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = F.expand_dims(x_embed, -1) / dim_t
        pos_y = F.expand_dims(y_embed, -1) / dim_t
        pos_x = F.flatten(F.stack(
            (F.sin(pos_x[:, :, :, 0::2]), F.cos(pos_x[:, :, :, 1::2])), axis=4), start_axis=3)
        pos_y = F.flatten(F.stack(
            (F.sin(pos_y[:, :, :, 0::2]), F.cos(pos_y[:, :, :, 1::2])), axis=4), start_axis=3)
        pos = F.transpose(F.concat((pos_y, pos_x), axis=3), (0, 3, 1, 2))
        return pos


def get_valid_boxes(raw_boxes, terminate, ignore):
    '''
    Input:
        raw_boxes: (B, MAXN, 4+1)
        data_tensor: (B, C, H, W)
    Return:
        boxes: list of (Nb, 4)
        labels:  list of (Nb,)
    '''
    # (B,)
    B = raw_boxes.shape[0]
    nr_valid_boxes = (1 - F.equal(raw_boxes[:, :, -1], terminate)).sum(axis=1).astype('int32')

    #print(f'nr_valid_boxes: {nr_valid_boxes}')

    # NOTE(peizhen): raw_boxes[i, :0, :4] will cause bug since ':0' indexing is invalid in megengine
    #boxes = [raw_boxes[i, :nr_valid_boxes[i], :4] for i in range(B)]
    #labels = [raw_boxes[i, :nr_valid_boxes[i], 4] for i in range(B)]

    # B x (Nb, 4) and B x (Nb,)
    boxes = list()
    labels = list()
    for i in range(B):
        num_valid = nr_valid_boxes[i].item()
        if num_valid > 0:
            boxes.append(raw_boxes[i, :num_valid, :4])
            labels.append(raw_boxes[i, :num_valid, 4])
        else:
            boxes.append(F.zeros((0, 4), dtype=raw_boxes.dtype, device=raw_boxes.device))
            labels.append(F.zeros((0,), dtype=raw_boxes.dtype, device=raw_boxes.device))

    # TODO(peizhen): currently discard those terms whose labels are -1. Need better operation ?
    # see backup/utility.py annotation part
    return boxes, labels


def get_instance_list(image_size, gt_boxes_human, gt_boxes_car, terminate=-2, ignore=-1):
    '''
    Input:
        gt_boxes_human: (B, MAXN, 4+1)
        gt_boxes_car:   (B, MAXN, 4+1)
    '''
    human_box_list, human_label_list = get_valid_boxes(gt_boxes_human, terminate, ignore)
    vehicle_box_list, vehicle_label_list = get_valid_boxes(gt_boxes_car, terminate, ignore)
    # (1) For `gt_boxes_human`, 1 denotes human. -2 denote invalid object determination (will be process as 0)
    # (2) For `gt_boxes_car`,   1 & 2 denotes different kinds of car, -2 denote invalid object determination (will still be 1 and 2)

    instances = list()
    contain_box_mask = list()
    nr_actual_boxes_per_img = list()
    for human_boxes, human_labels, vehicle_boxes, vehicle_labels in \
            zip(human_box_list, human_label_list, vehicle_box_list, vehicle_label_list):
        # (k, 4) and (k,)
        gt_boxes = F.concat([human_boxes, vehicle_boxes], axis=0).astype("float32")
        # Process gt_boxes_human's labels from 1 to 0. Naturally, car owns label 1 and 2
        gt_classes = F.concat([human_labels - 1, vehicle_labels], axis=0).astype("int32")

        contain_box_mask.append(gt_boxes.shape[0] > 0)
        assert gt_boxes.shape[0] == gt_classes.shape[0]

        # pad box for images who contain no any box to work around potential indexing bug (unlike in coco, an image in business dataset might contain no any image)
        nr_valid_objs = gt_boxes.shape[0]
        nr_actual_boxes_per_img.append(nr_valid_objs)
        if nr_valid_objs == 0:
            gt_boxes = F.zeros((1, 4), device=gt_boxes.device, dtype=gt_boxes.dtype)
            gt_classes = F.zeros((1,), device=gt_classes.device, dtype=gt_classes.dtype)

        instances.append({'image_size': image_size, 'gt_boxes': gt_boxes, 'gt_classes': gt_classes})

    # (bsz,)
    contain_box_mask = mge.Tensor(
        contain_box_mask, device=instances[0]['gt_boxes'].device, dtype='float32').detach()

    return instances, contain_box_mask, nr_actual_boxes_per_img
