import math
from mmdet3d.models.detectors.base import Base3DDetector
import torch
from torch import nn as nn

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS, build_backbone, build_neck, build_head


@DETECTORS.register_module()
class FVNet(Base3DDetector):

    def __init__(self,
                 projection_cfg=None,
                 fv_backbone=None,
                 fv_neck=None,
                 img_backbone=None,
                 img_neck=None,
                 bbox_head=None,
                 fusion_mode=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(FVNet, self).__init__(init_cfg)

        self.projection_cfg = projection_cfg
        if fv_backbone:
            self.fv_backbone = build_backbone(fv_backbone)
        if fv_neck:
            self.fv_neck = build_neck(fv_neck)
        if img_backbone:
            self.img_backbone = build_backbone(img_backbone)
        if img_neck:
            self.img_neck = build_neck(img_neck)
        if bbox_head:
            self.bbox_head = build_head(bbox_head)

        self.fusion_mode = fusion_mode
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # TODO: image backbone pretrained 구현

    @property
    def with_fv_backbone(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'fv_backbone') and self.fv_backbone is not None

    @property
    def with_fv_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'fv_neck') and self.fv_neck is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_fv_feat(self, fv, img_metas):
        """Extract features of points."""
        if self.with_fv_backbone and fv is not None:
            x = self.fv_backbone(fv)
        else:
            return None
        if self.with_fv_neck:
            x = self.fv_neck(x)
        return x

    def extract_feat(self, fv, img, img_metas):
        fv_feats = self.extract_fv_feat(fv, img_metas)
        img_feats = self.extract_img_feat(img, img_metas)
        return (fv_feats, img_feats)

    def get_valid_coords(self, fv):
        valid_coords = dict()
        b, v, u = torch.nonzero(fv[:, -1, :, :], as_tuple=True)
        valid_coords_2d = torch.stack([b, v, u]).T
        valid_coords_3d = fv[b, :3, v, u]
        valid_coords_3d = torch.cat([b.unsqueeze(dim=-1), valid_coords_3d], dim=1)
        valid_coords['2d'] = valid_coords_2d
        valid_coords['3d'] = valid_coords_3d

        return valid_coords
    
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
                               (pts_2d[i, 1, :] < height) & (pts_2d[i, 1, :] >= 0))[0]
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

    def fusion(self, fv_feats, img_feats, fusion_mode):
        if fusion_mode is None:
            if fv_feats is not None:
                return fv_feats
            if img_feats is not None:
                return img_feats

    def resize_and_pad(self, fv_list, img_metas):
        # Resize
        new_fv_list = []
        for i in range(len(fv_list)):
            # Resize
            fv = fv_list[i]
            w_src, h_src = fv.shape[2], fv.shape[1]
            w_des, h_des = self.projection_cfg['size']
            w_scale, h_scale = (w_des / w_src, h_des / h_src)
            if not (w_scale == 1. and h_scale == 1.):
                fv_resized = torch.zeros((fv.shape[0], h_des, w_des),
                                          dtype=fv.dtype,
                                          device=fv.device)
                _, src_v, src_u = torch.nonzero(fv, as_tuple=True)
                src_idx = torch.stack([src_v, src_u]).unique(dim=1)
                des_v = (src_idx[0] * h_scale).to(torch.long)
                des_u = (src_idx[1] * w_scale).to(torch.long)
                fv_resized[:, des_v, des_u] = fv[:, src_idx[0, :], src_idx[1, :]]
                fv = fv_resized
            # Padding
            divisor = self.projection_cfg['divisor']
            h_src, w_src = fv.shape[1], fv.shape[2]
            h_des = math.ceil(fv.shape[1] / divisor) * divisor
            w_des = math.ceil(fv.shape[2] / divisor) * divisor
            h_pad = h_des - h_src
            w_pad = w_des - w_src
            if not (h_pad == 0 and w_pad == 0):
                fv_padded = torch.zeros((fv.shape[0], h_des, w_des),
                                         dtype=fv.dtype,
                                         device=fv.device)
                fv_padded[:, :h_src, :w_src] = fv
                fv = fv_padded
            new_fv_list.append(fv) 
        fv = torch.stack(new_fv_list)
        return fv
    
    def filter_valid_feats(self, feats, valid_coords):
        # feats: list [(B, 64, h, w)]
        # valid_coords['2d']: torch.Tenosr (N, 3)
        # valid_coords['3d']: torch.Tensor (N, 4)
        # h*w 개의 픽셀 중 N개 필터링
        # TODO: multi scale 구현(get valid coords먼저)

        b = valid_coords['2d'][:, 0]
        v = valid_coords['2d'][:, 1]
        u = valid_coords['2d'][:, 2]
        valid_feats = feats[0][b, :, v, u] # list [(N, 64)]

        # concat xyz
        xyz = valid_coords['3d'][:, 1:]
        valid_feats = torch.cat([xyz, valid_feats], dim=1)

        return [valid_feats]

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      imgs=None,
                      gt_bboxes_ignore=None):

        fv = self.pts_to_fv(points, img_metas) # list B x (5, h, w)
        fv = self.resize_and_pad(fv, img_metas) # (B, 5, h', w')

        # flip
        p = torch.rand(1)[0]
        if p > 0.5:
            fv = torch.flip(fv, dims=[3])
            fv[:, 1, :, :] *= -1
            for i in range(len(gt_bboxes_3d)):
                gt_bboxes_3d[i].tensor[:, 1] *= -1
        # # scale
        scale_factor = torch.rand(1).item() * 0.1 + 0.95
        fv[:, :3, :, :] *= scale_factor
        for i in range(len(gt_bboxes_3d)):
            gt_bboxes_3d[i].tensor[:, :6] *= scale_factor

        valid_coords = self.get_valid_coords(fv)
        fv_feats, img_feats = self.extract_feat(fv, imgs, img_metas)
        feats = self.fusion(fv_feats, img_feats, self.fusion_mode)
        pts_feats = self.filter_valid_feats(feats, valid_coords)
        outs = self.bbox_head(pts_feats)
        losses = self.bbox_head.loss(*outs,
                                     gt_bboxes_3d,
                                     gt_labels_3d,
                                     valid_coords)
        return losses
    
    def simple_test(self,
                    points,
                    img_metas,
                    imgs=None,
                    gt_bboxes_3d=None,
                    gt_labels_3d=None,
                    rescale=False):
        """Test function without augmentaiton."""
        """Implemented only for batch size of one."""
        points = [points]
        img_metas = [img_metas]
        fv = self.pts_to_fv(points, img_metas) # list B x (5, h, w)
        fv = self.resize_and_pad(fv, img_metas) # (B, 5, h', w')
        valid_coords = self.get_valid_coords(fv)
        fv_feats, img_feats = self.extract_feat(fv, imgs, img_metas)
        feats = self.fusion(fv_feats, img_feats, self.fusion_mode)
        pts_feats = self.filter_valid_feats(feats, valid_coords)
        outs = self.bbox_head(pts_feats)
        bbox_list = self.bbox_head.get_bboxes(*outs,
                                              img_metas,
                                              valid_coords,
                                              gt_bboxes_3d,
                                              gt_labels_3d,
                                              rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]
