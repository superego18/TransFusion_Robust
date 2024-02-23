import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from .. import builder
from .mvx_two_stage import MVXTwoStageDetector

import numpy as np

@DETECTORS.register_module()
class TransFusionDetector(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self, **kwargs):
        super(TransFusionDetector, self).__init__(**kwargs)

        self.freeze_img = kwargs.get('freeze_img', True)
        self.init_weights(pretrained=kwargs.get('pretrained', None))

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        super(TransFusionDetector, self).init_weights(pretrained)

        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img.float())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    # add for robust
    # one implemetation for one result
    def show_results_for_robust(self, gt_info, classes, data, result, show_detail, out_dir):
        """Results visualization.

        Args:
            data (dict): Input points and the information of the sample. {points:, }
            result (dict): Prediction results. (pred_bboxes_3d, pred_labels_3d, pred_scores_3d) ## [{}, {}]
            out_dir (str): Output directory of visualization result.
        """
        
        for batch_id in range(len(result)):
            
            print('batch_id:', batch_id)
            
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} "
                           f'for visualization!')
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id][
                    'pts_filename']
                # box_mode_3d = data['img_metas'][0]._data[0][batch_id][
                #     'box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            file_name = osp.split(pts_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'
            
            #TODO: Set a threshold of pred_bboxes_3d's scores
            inds = result[batch_id]['pts_bbox']['scores_3d'] > 0.1
            pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d'][inds]
            
            # for now we convert points and bbox into depth mode
            # if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d== Box3DMode.LIDAR):
            #     points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR, Coord3DMode.DEPTH)
            #     pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d, Box3DMode.DEPTH)
            # elif box_mode_3d != Box3DMode.DEPTH:
                # ValueError(f'Unsupported box_mode_3d {box_mode_3d} for convertion!')
                
            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            pred_bboxes[:, [-1]] *= -1 # to c.w
            
            ## add pred_labels
            pred_labels_ = result[batch_id]['pts_bbox']['labels_3d'][inds].cpu().numpy()
            pred_scores = result[batch_id]['pts_bbox']['scores_3d'][inds].cpu().numpy()
            
            # pred_labels = []
            # for label, score in zip(pred_labels_, pred_scores):
            #     pred_labels.append(CLASSES[label]+': '+str(round(score, 3)))
            # pred_labels = np.array(pred_labels)
            
            pred_labels = np.array([classes[label] for label in pred_labels_])
            
            gt_bboxes = gt_info['annos']['gt_bboxes']
            gt_bboxes[:, [-1]] *= -1 # to c.w
            gt_labels = gt_info['annos']['name']
            
            sample_idx = data['img_metas'][0].data[0][0]['sample_idx']
        
            print()
            print('-'*100)
            print(f'This is visualization of result of {sample_idx}')
            print('-'*100)
            print()
        
            imgs = data['img_metas'][0]._data[0][batch_id]['filename']
            lidar2img = data['img_metas'][0]._data[0][batch_id]['lidar2img']
            
            if show_detail:
                clss = set(pred_labels).union(set(gt_labels))
                
                from mmdet3d.core.bbox import box_np_ops as box_np_ops
                import cv2
                
                for cls in clss:
                    print(f'[[[{cls}]]]')
                    pred_cls_inds = (pred_labels == cls)
                    pred_bboxes_ = pred_bboxes[pred_cls_inds]
                    
                    gt_cls_inds = (gt_labels == cls)
                    gt_bboxes_ = gt_bboxes[gt_cls_inds] 
                    
                    
                    point_indices_gt = box_np_ops.points_in_rbbox(points, gt_bboxes_, origin=(0.5, 0.5, 0))
                    point_indices_gt = [any(row) for row in point_indices_gt]
                    points_cls_gt = points[point_indices_gt][:,:3]
                    
                    point_indices = box_np_ops.points_in_rbbox(points, pred_bboxes_, origin=(0.5, 0.5, 0))
                    point_indices = [any(row) for row in point_indices]
                    points_cls_ = points[point_indices][:,:3]
                    
                    
                    cams = ['front', 'left', 'rear', 'right']
                    
                    for c in range(len(cams)):
                        
                        if c == 0: # front
                            indices = np.where(points_cls_[:, 0] >= 0)
                            indices_gt = np.where(points_cls_gt[:, 0] >= 0)
                            img_points = points_cls_[indices]
                            img_points_gt = points_cls_gt[indices_gt]
                        elif c == 1: # left
                            indices = np.where(points_cls_[:, 1] >= 0)
                            indices_gt = np.where(points_cls_gt[:, 1] >= 0)
                            img_points = points_cls_[indices]
                            img_points_gt = points_cls_gt[indices_gt]
                        elif c == 2: # rear
                            indices = np.where(points_cls_[:, 0] <= 0)
                            indices_gt = np.where(points_cls_gt[:, 0] <= 0)
                            img_points = points_cls_[indices]
                            img_points_gt = points_cls_gt[indices_gt]
                        elif c == 3: # right
                            indices = np.where(points_cls_[:, 1] <= 0)
                            indices_gt = np.where(points_cls_gt[:, 1] <= 0)
                            img_points = points_cls_[indices]
                            img_points_gt = points_cls_gt[indices_gt]
                    
                        img_points = np.dot(lidar2img[c], np.hstack((img_points, np.ones((img_points.shape[0], 1)))).T)
                        img_points = img_points.T / img_points.T[:, 2][:, np.newaxis]
                        
                        img_points_gt = np.dot(lidar2img[c], np.hstack((img_points_gt, np.ones((img_points_gt.shape[0], 1)))).T)
                        img_points_gt = img_points_gt.T / img_points_gt.T[:, 2][:, np.newaxis]
                        
                        img = cv2.imread(imgs[c])
                        
                        
                        for i in range(img_points_gt.shape[0]):
                            x, y = int(img_points_gt[i, 0]), int(img_points_gt[i, 1])
                            # 이미지에 포인트 그리기
                            cv2.circle(img, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

                        # 이미지 좌표로 변환된 포인트 그리기
                        for i in range(img_points.shape[0]):
                            x, y = int(img_points[i, 0]), int(img_points[i, 1])
                            # 이미지에 포인트 그리기
                            cv2.circle(img, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
     
                        # 추가할 텍스트
                        text = f'{sample_idx}/{cams[c]}/{cls}/gt:{(gt_bboxes_.shape[0])}/pred:{pred_bboxes_.shape[0]}'
                        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

                        # 이미지 보여주기
                        cv2.imshow('Image with Lidar Points', img)
                        cv2.waitKey(0)

                    cv2.destroyAllWindows()

                    show_result(points, gt_bboxes_, pred_bboxes_, out_dir, file_name)
                    
                
            else:
                show_result(points, gt_bboxes, pred_bboxes, out_dir, file_name)
