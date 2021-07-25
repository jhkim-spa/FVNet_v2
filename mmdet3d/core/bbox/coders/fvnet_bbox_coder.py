import torch
import numpy as np

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class FVNetBBoxCoder(BaseBBoxCoder):
    """Bbox Coder for 3D boxes.

    Args:
        code_size (int): The dimension of boxes to be encoded.
    """
    # TODO: sine difference

    def __init__(self, prior_size=[1.6, 3.9, 1.56], code_size=7):
        super(FVNetBBoxCoder, self).__init__()
        self.prior_size = prior_size
        self.code_size = code_size

    @staticmethod
    def encode(points, boxes, prior_size):
        # normalize
        # size = torch.tensor(prior_size)
        # wa, la, ha = size
        # diagonal = torch.sqrt(wa**2 + la**2)
        # xt = (boxes[:, 0] - points[:, 0]) / diagonal
        # yt = (boxes[:, 1] - points[:, 1]) / diagonal
        # zt = (boxes[:, 2] + boxes[:, 5]/2 - points[:, 2]) / ha
        # wt = torch.log(boxes[:, 3] / prior_size[0])
        # lt = torch.log(boxes[:, 4] / prior_size[1])
        # ht = torch.log(boxes[:, 5] / prior_size[2])
        # rt_cos = torch.cos(boxes[:, 6])
        # rt_sin = torch.sin(boxes[:, 6])

        # not normalize
        xt = boxes[:, 0] - points[:, 0]
        yt = boxes[:, 1] - points[:, 1]
        zt = boxes[:, 2] + boxes[:, 5]/2 - points[:, 2]
        wt = torch.log(boxes[:, 3])
        lt = torch.log(boxes[:, 4])
        ht = torch.log(boxes[:, 5])
        rt_cos = torch.cos(boxes[:, 6])
        rt_sin = torch.sin(boxes[:, 6])

        return torch.stack([xt, yt, zt, wt, lt, ht, rt_cos, rt_sin]).T

    @staticmethod
    def decode(points, deltas, prior_size):
        # normalize
        # size = torch.tensor(prior_size)
        # wa, la, ha = size
        # diagonal = torch.sqrt(wa**2 + la**2)
        # xg = points[:, 0] + deltas[:, 0] * diagonal
        # yg = points[:, 1] + deltas[:, 1] * diagonal
        # zg = points[:, 2] + deltas[:, 2] * ha

        # wg = torch.exp(deltas[:, 3]) * prior_size[0]
        # lg = torch.exp(deltas[:, 4]) * prior_size[1]
        # hg = torch.exp(deltas[:, 5]) * prior_size[2]
        # zg = zg - hg / 2
        # rg = torch.atan2(deltas[:, 7], deltas[:, 6])

        # not normalize
        xg = points[:, 0] + deltas[:, 0]
        yg = points[:, 1] + deltas[:, 1]
        zg = points[:, 2] + deltas[:, 2]

        wg = torch.exp(deltas[:, 3])
        lg = torch.exp(deltas[:, 4])
        hg = torch.exp(deltas[:, 5])
        zg = zg - hg / 2
        rg = torch.atan2(deltas[:, 7], deltas[:, 6])
        return torch.stack([xg, yg, zg, wg, lg, hg, rg]).T
