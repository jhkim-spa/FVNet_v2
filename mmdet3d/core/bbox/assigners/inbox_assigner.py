import torch

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu



@BBOX_ASSIGNERS.register_module()
class InBoxAssigner(BaseAssigner):

    def assign(self, anchors, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):

        points = anchors[:, :3].clone()
        points[:, 2] += anchors[:, 5] / 2
        num_points = points.shape[0]
        num_gts = gt_bboxes.shape[0]

        if num_gts == 0 or num_points == 0:
            # If no truth assign everything to the background
            assigned_gt_inds = points.new_full((num_points, ),
                                               0,
                                               dtype=torch.long)
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = points.new_full((num_points, ),
                                                  -1,
                                                  dtype=torch.long)
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        gt_bboxes = gt_bboxes.unsqueeze(dim=0)
        points = points.unsqueeze(dim=0)
        assigned_gt_inds = points_in_boxes_gpu(points, gt_bboxes).to(torch.long).squeeze(dim=0)
        assigned_gt_inds = assigned_gt_inds + 1

        # stores the assigned gt index of each point

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_points, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
