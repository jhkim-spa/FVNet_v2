import numpy as np
import torch
from mmcv.runner import force_fp32
from torch import nn as nn

from mmdet3d.core import (PseudoSampler, box3d_multiclass_nms,
                          limit_period, xywhr2xyxyr)
from mmdet.core import build_assigner, build_bbox_coder, build_sampler, multi_apply
from mmdet.models import HEADS
from ..builder import build_loss
from .anchor3d_head import Anchor3DHead


@HEADS.register_module()
class FVNetHead(nn.Module):

    def __init__(self,
                 num_classes,
                 feat_channels=64,
                 bbox_coder=dict(type='FVNetBBoxCoder'),
                 fg_weight=1,
                 bg_weight=1,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.feat_channels = feat_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # build box coder
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.box_code_size = self.bbox_coder.code_size

        # build loss function
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        self._init_layers()

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        # Classification layer
        self.cls_fc = nn.Sequential(
            nn.Linear(self.feat_channels, self.num_classes)
        )
        # Regression layer
        self.reg_fc = nn.Sequential(
            nn.Linear(self.feat_channels, self.box_code_size)
        )

    def forward_single(self, x):
        cls_score = self.cls_fc(x)
        bbox_pred = self.reg_fc(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             valid_coords):

        cls_reg_targets = self.get_targets(
            valid_coords,
            gt_bboxes,
            gt_labels)

        (cls_targets, bbox_targets, pos_idx,
         num_total_list, num_pos_list) = cls_reg_targets

        losses_cls, losses_loc, losses_size, losses_angle = multi_apply( # multi-scale
            self.loss_single,
            [valid_coords], # TODO: multi-scale
            cls_scores,
            bbox_preds,
            [cls_targets],
            [bbox_targets],
            [pos_idx],
            [num_total_list],
            [num_pos_list])

        return dict(
            loss_cls=losses_cls, loss_loc=losses_loc,
            loss_size=losses_size, loss_angle=losses_angle)

    def loss_single(self, valid_coords, cls_score, bbox_pred,
                    cls_target, bbox_target, pos_idx,
                    num_total_list, num_pos_list):
        num_pos = sum(num_pos_list)
        # classification loss
        cls_weight = torch.ones_like(cls_target) * self.bg_weight
        cls_weight[cls_target != 1] = self.fg_weight
        loss_cls = self.loss_cls(cls_score,
                                 cls_target,
                                 weight=cls_weight,
                                 reduction_override='sum')
        loss_cls = loss_cls / cls_score.shape[0]
        # regression loss
        if num_pos > 0:
            bbox_pred = bbox_pred.split(num_total_list)
            pos_idx = pos_idx.split(num_pos_list)
            pos_bbox_pred = [pred[idx] for pred, idx in zip(bbox_pred, pos_idx)]
            pos_bbox_pred = torch.cat(pos_bbox_pred)
            # loss_bbox = self.loss_bbox(pos_bbox_pred,
            #                            bbox_target,
            #                            avg_factor=num_pos)
            loss_bbox = self.loss_bbox(pos_bbox_pred,
                                       bbox_target,
                                       reduction_override='none')
            loc_loss = loss_bbox[:, :3].sum() / num_pos
            size_loss = loss_bbox[:, 3:6].sum() / num_pos
            angle_loss = loss_bbox[:, 6:].sum() / num_pos
        else:
            loc_loss = bbox_pred.sum() * 0
            size_loss = bbox_pred.sum() * 0
            angle_loss = bbox_pred.sum() * 0


        # import matplotlib.pyplot as plt
        # points = valid_coords['3d'][:, 1:].cpu()
        # targets = valid_coords['3d'][cls_target==0][:, 1:].cpu()
        # w, l, h = torch.tensor([1.6, 3.9, 1.56])
        # diagonal = torch.sqrt(w**2 + l**2)
        # x = points[cls_target==0][:, 0] + bbox_target[:, 0].cpu() * diagonal
        # y = points[cls_target==0][:, 1] + bbox_target[:, 1].cpu() * diagonal

        # plt.scatter(points[:, 1], points[:, 0], s=0.01, color='k')
        # plt.scatter(targets[:, 1], targets[:, 0], s=0.02, color='red')
        # plt.scatter(y, x, s=0.02, color='blue')
        # plt.gca().set_aspect('equal')
        # plt.savefig('test.png')

        # print('test')

        return loss_cls, loc_loss, size_loss, angle_loss

    def get_targets(self, valid_coords, gt_bboxes, gt_labels):
        """
        Args: valid_coords['2d']: (23720, 3)   N * (b, v, u)
              valid_coords['3d']: (23720, 4)   N * (b, x, y, z)
              gt_bboxes: list len==batch_size 
                         gt_bboxes[0]: LiDARInstance3DBoxes
                         gt_bboxes[0].tensor.shape: (N, 7) N: num of gts
              gt_labels: list len==batch_size
                         gt_labels[0].shape: (N) fiiled with -1, 0
        """ 
        from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu
        # cls targets 
        batch_size = valid_coords['2d'][-1, 0].item() + 1
        device = gt_labels[0].device
        valid_idx = [labels != -1 for labels in gt_labels]
        gt_bboxes = [boxes[idx] for boxes, idx in zip(gt_bboxes, valid_idx)]
        gt_labels = [labels[idx] for labels, idx in zip(gt_labels, valid_idx)]

        cls_targets_list = []
        bbox_targets_list = []
        pos_idx_list = []
        num_total_list = []
        num_pos_list = []
        total_pos = 0
        for i in range(batch_size):
            idx = valid_coords['3d'][:, 0] == i
            boxes = gt_bboxes[i].tensor
            boxes = boxes.to(device)
            points = valid_coords['3d'][idx][:, 1:]
            # check points in gt boxes
            assigned_idx = points_in_boxes_gpu(points.unsqueeze(0),
                boxes.unsqueeze(0))[0].to(torch.long)
            pos_idx = torch.where(assigned_idx != -1)[0]

            neg_idx = torch.where(assigned_idx == -1)[0]
            num_total = points.shape[0]
            num_pos = pos_idx.shape[0]

            cls_targets = torch.zeros((points.shape[0]),
                                       dtype=torch.long, device=device)
            cls_targets[neg_idx] = 1

            if boxes.shape[0] != 0:
                boxes = boxes[assigned_idx][pos_idx]
                bbox_targets = self.bbox_coder.encode(points[pos_idx], boxes,
                    self.bbox_coder.prior_size)
            cls_targets_list.append(cls_targets)
            num_total_list.append(num_total)
            num_pos_list.append(num_pos)
            total_pos += num_pos
            if num_pos != 0:
                bbox_targets_list.append(bbox_targets)
                pos_idx_list.append(pos_idx)
        
        cls_targets = torch.cat(cls_targets_list, dim=0)
        if len(pos_idx_list) != 0:
            pos_idx = torch.cat(pos_idx_list, dim=0)
        else:
            pos_idx = torch.tensor([], device=device)
        if total_pos != 0:
            bbox_targets = torch.cat(bbox_targets_list, dim=0)
        else:
            bbox_targets = None

        return (cls_targets, bbox_targets, pos_idx,
                num_total_list, num_pos_list)

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   input_metas,
                   valid_coords,
                   gt_bboxes=None,
                   gt_labels=None,
                   rescale=False):

        # TODO: multi-sclae 구현
        cfg = self.test_cfg

        points = valid_coords['3d'][:, 1:]
        cls_scores = cls_scores[0]
        bbox_preds = bbox_preds[0]
        input_metas = input_metas[0]
        scores =cls_scores.sigmoid()

        # Remove False Positive
        # from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu
        # valid_idx = gt_labels[0] == 0
        # boxes = gt_bboxes[0].tensor[valid_idx].to(points.device)
        # assigned_idx = points_in_boxes_gpu(points.unsqueeze(0),
        #     boxes.unsqueeze(0))[0].to(torch.long)
        # pos_inds = assigned_idx != -1

        # scores_tmp = scores.clone()
        # scores *= 0
        # scores[pos_inds] = scores_tmp[pos_inds]

        # Visualize segmentation results
        # self.visualize(valid_coords, gt_bboxes, gt_labels,
        #                 points, scores, input_metas, bev_seg=True, fv_seg=True)

        nms_pre = cfg.get('nms_pre', -1)
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            max_scores, _ = scores.max(dim=1)
            _, topk_inds = max_scores.topk(nms_pre)
            bbox_preds = bbox_preds[topk_inds, :]
            scores = scores[topk_inds, :]
            points = points[topk_inds, :]
        
        bboxes = self.bbox_coder.decode(points, bbox_preds, self.bbox_coder.prior_size)
        bboxes_for_nms = xywhr2xyxyr(input_metas['box_type_3d'](
            bboxes, box_dim=7).bev)

        padding = scores.new_zeros(scores.shape[0], 1)
        scores = torch.cat([scores, padding], dim=1)
 
        score_thr = cfg.get('score_thr', 0)
        results = box3d_multiclass_nms(bboxes, bboxes_for_nms, scores,
                                       score_thr, cfg.max_num, cfg)
        bboxes, scores, labels = results
        bboxes = input_metas['box_type_3d'](bboxes, box_dim=7)
        proposals = (bboxes, scores, labels)

        return [proposals]

    def pts_to_fv(self, points, img_metas):
        # TODO: Flip 어떻게 처리??
        device = points[0].device
        lidar2img = [torch.tensor(res['lidar2img'], device=device) for res\
            in img_metas]
        lidar2img = torch.stack(lidar2img) # (B, 4, 4)
        num_points = [pts.shape[0] for pts in points]
        for i in range(len(points)):
            pts = points[i]
            if pts.shape[0] < max(num_points):
                res = max(num_points) - pts.shape[0]
                points[i] = torch.cat([pts, pts.new_zeros(res, pts.shape[1])], 0)
        points = torch.stack(points) # (B, max(N), 4)

        reflectances = points[..., -1]
        points = points[..., :3] # (B, max(N), 3)
        proj_velo2cam2 = lidar2img[:, :3, :]
        pts_2d = self.project_to_image(points.permute(0, 2, 1), proj_velo2cam2)

        fv_list = []
        for i in range(points.shape[0]):
            width, height = img_metas[i]['img_info']['img_shape']
            inds = torch.where((pts_2d[i, 0, :] < width) & (pts_2d[i, 0, :] >= 0) &
                               (pts_2d[i, 1, :] < height) & (pts_2d[i, 1, :] >= 0) &
                               (points[i, :, 0] > 0))[0]
            imgfov_pc_pixel = pts_2d[i, :, inds]
            imgfov_pc_velo = points[i, inds, :]
            reflectance = reflectances[i, inds]

            fv = points[0].new_zeros(height, width, 5)
            x_coords = torch.trunc(imgfov_pc_pixel[0]).to(torch.long)
            y_coords = torch.trunc(imgfov_pc_pixel[1]).to(torch.long)
            fv[y_coords, x_coords, :3] = imgfov_pc_velo
            fv[y_coords, x_coords, 3] = reflectance
            flag_channel = (fv[:, :, 0] > 0)
            fv[:, :, -1] = flag_channel
            fv = fv.permute(2, 0, 1)
            fv_list.append(fv)

        return fv_list

    def project_to_image(self, points, proj_mat):
        # points (B, 3, max(N))
        # proj_mat (B, 3, 4)
        batch_size = points.shape[0]
        num_pts = points.shape[2]

        points = torch.cat([points, points.new_ones(batch_size, 1, num_pts)], 1)
        points = torch.bmm(proj_mat, points) # (B, 3, 4) @ (B, 4, max(N))
        # points (B, 3, max(N))
        points[:, :2, :] /= points[:, 2, :].reshape(points.shape[0], 1, points.shape[2])
        return points[:, :2, :] #(B, 2, max(N))
    
    def get_valid_coords(self, fv):
        valid_coords = dict()
        b, v, u = torch.nonzero(fv[:, 0, :, :], as_tuple=True)
        valid_coords_2d = torch.stack([b, v, u]).T
        valid_coords_3d = fv[b, :3, v, u]
        valid_coords_3d = torch.cat([b.unsqueeze(dim=-1), valid_coords_3d], dim=1)
        valid_coords['2d'] = valid_coords_2d
        valid_coords['3d'] = valid_coords_3d

        return valid_coords
    
    def visualize(self,
                  valid_coords,
                  gt_bboxes,
                  gt_labels,
                  points,
                  scores,
                  input_metas,
                  bev_seg=True,
                  fv_seg=True):
        import matplotlib.pyplot as plt
        import os
        cls_reg_targets = self.get_targets(
            valid_coords,
            gt_bboxes,
            gt_labels)
        pos_idx = cls_reg_targets[2]
        if pos_idx.shape[0] != 0:
            gt_points = points[pos_idx].cpu()
            img_name = input_metas['img_info']['filename'].split('/')[-1]

            threshold = 0.7
            if bev_seg:
                if not os.path.exists('plots/seg_bev'):
                    os.makedirs('plots/seg_bev')
                plt.scatter(points[:, 1].cpu(), points[:, 0].cpu(), s=0.01)
                plt.scatter(gt_points[:, 1],
                            gt_points[:, 0], s=0.2, color='r')
                plt.scatter(points[:, 1].cpu() + 50, points[:, 0].cpu(), s=0.01)
                idx = scores > threshold
                fg_points = points[idx.reshape(-1)]
                plt.scatter(fg_points[:, 1].cpu() + 50, fg_points[:, 0].cpu(), s=0.2, color='r')
                plt.xlim(-30, 80)
                plt.ylim(0, 70)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.savefig(f"plots/seg_bev/{img_name}")
                plt.close()

            if fv_seg:
                if not os.path.exists('plots/seg_fv'):
                    os.makedirs('plots/seg_fv')
                fv = self.pts_to_fv([points], [input_metas]) 
                fv_gt = self.pts_to_fv([gt_points], [input_metas])
                idx = scores > threshold
                fv_pred = self.pts_to_fv([points[idx.reshape(-1)]], [input_metas])

                import matplotlib.pyplot as plt
                fv_2d = self.get_valid_coords(fv[0].unsqueeze(dim=0))['2d'][:, 1:]
                gt_2d = self.get_valid_coords(fv_gt[0].unsqueeze(dim=0))['2d'][:, 1:]
                pred_2d = self.get_valid_coords(fv_pred[0].unsqueeze(dim=0))['2d'][:, 1:]

                plt.scatter(fv_2d[:, 1].cpu(), -fv_2d[:, 0].cpu(), s=0.001, color='k')
                plt.scatter(gt_2d[:, 1].cpu(), -gt_2d[:, 0].cpu(), s=0.01, color='r')
                plt.scatter(fv_2d[:, 1].cpu(), -fv_2d[:, 0].cpu()-380, s=0.001, color='k')
                plt.scatter(pred_2d[:, 1].cpu(), -pred_2d[:, 0].cpu()-380, s=0.01, color='r')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.xlim(0, 1300)
                plt.ylim(-800, 0)
                plt.savefig(f"plots/seg_fv/{img_name}")
                plt.close()
    