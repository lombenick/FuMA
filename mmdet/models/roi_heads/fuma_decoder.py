import mmcv
import torch
from typing import List, Optional, Tuple
from mmdet.core import bbox2result, bbox_xyxy_to_cxcywh
from mmdet.core.bbox.samplers import PseudoSampler
from ..builder import HEADS
from .cascade_roi_head import CascadeRoIHead


@HEADS.register_module()
class FuMADecoder(CascadeRoIHead):
    def __init__(self,
                 num_stages: int = 6,
                 stage_loss_weights: Tuple = (1, 1, 1, 1, 1, 1),
                 query_dim: int = 256,
                 featmap_strides: List[int]= [4, 8, 16, 32],
                 bbox_head: Optional[mmcv.ConfigDict] = None,
                 train_cfg: Optional[mmcv.ConfigDict] = None,
                 test_cfg: Optional[mmcv.ConfigDict] = None,
                 pretrained: Optional[str] = None,
                 init_cfg: Optional[mmcv.ConfigDict]=None):
        assert bbox_head is not None
        assert len(stage_loss_weights) == num_stages
        self.featmap_strides = featmap_strides
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.query_dim = query_dim
        super(FuMADecoder, self).__init__(
            num_stages,
            stage_loss_weights,
            bbox_roi_extractor=dict(
                # This does not mean that our method need RoIAlign. We put this
                # as a placeholder to satisfy the argument for the parent class
                # 'CascadeRoIHead'.
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(self.bbox_sampler[stage], PseudoSampler)

    def _bbox_forward(self,
                      stage: int, 
                      mlvl_feats: Tuple[torch.Tensor],
                      query_feats: torch.Tensor, 
                      bboxes: torch.Tensor,
                      featmap_strides: List[int],
                      img_metas: List[dict]):
        '''
        args:
            query_feats: (Tensor) shape (batch_size, num_query, embed_dims)
            bboxes: (Tensor) shape (batch_size, num_query, 4),last dimension is 
                (tl_x, tl_y, br_x, br_y)(real coordinate)
            feat_tokens: (Tensor) shape (batch_size, H*W, embed_dims)
        '''
        num_imgs = len(img_metas)
        bbox_head = self.bbox_head[stage]
        cls_score, delta_bbox, query_feats = bbox_head(mlvl_feats, 
            query_feats, bboxes, featmap_strides)
        
        bboxes, decoded_bboxes = self.bbox_head[stage].refine_xysr(bboxes, delta_bbox)
        bboxes_list = [item for item in decoded_bboxes]

        # 返回回归框
        bbox_results = dict(
            cls_score=cls_score,
            bboxes=bboxes,
            decode_bbox_pred=decoded_bboxes,
            query_feats=query_feats,
            detach_cls_score_list=[
                cls_score[i].detach() for i in range(num_imgs)
            ],
            detach_bboxes_list=[item.detach() for item in bboxes_list],
            bboxes_list=bboxes_list,
        )

        return bbox_results

    def forward_train(self,
                      mlvl_feats: Tuple[torch.Tensor],
                      bboxes: torch.Tensor,
                      query_feats: torch.Tensor,
                      img_metas: List[dict],
                      gt_bboxes: List[torch.Tensor],
                      gt_labels: List[torch.Tensor],
                      gt_bboxes_ignore: List[torch.Tensor] = None,
                      imgs_whwh: torch.Tensor = None,
                      gt_masks: torch.Tensor = None):
        '''
            args:
                mlvl_feats: (Tuple[Tensor]) the feature maps of the images, has 
                    shape (num_levels, batch_size, embed_dims, H, W)(Make the H 
                    and W of the feature map in a batch size the same after padding)
                bboxes: (Tensor) shape (batch_size, num_query, 4), last dimension 
                    (x1, y1, x2, y2)(real coordinate)
                query_feats: (Tensor) shape (batch_size, num_query, embed_dims)
                imgs_whwh: (Tensor) shape (batch_size, 1, 4) (real images)
        '''

        num_imgs = len(img_metas)
        imgs_whwh = imgs_whwh.repeat(1, query_feats.size(1), 1) # shape (batch_size, num_query, 4)
        all_stage_bbox_results = []
        all_stage_loss = {}
        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(stage, 
                                              mlvl_feats,
                                              query_feats, 
                                              bboxes,
                                              self.featmap_strides,
                                              img_metas)

            all_stage_bbox_results.append(bbox_results)
            if gt_bboxes_ignore is None:
                # TODO support ignore
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            cls_pred_list = bbox_results['detach_cls_score_list']
            bboxes_list = bbox_results['detach_bboxes_list']

            bboxes = bbox_results['bboxes'].detach()
            query_feats = bbox_results['query_feats']

            if self.stage_loss_weights[stage] <= 0:
                continue

            for i in range(num_imgs):
                normalize_bbox_ccwh = bbox_xyxy_to_cxcywh(bboxes_list[i] / imgs_whwh[i])
                assign_result = self.bbox_assigner[stage].assign(
                    normalize_bbox_ccwh, cls_pred_list[i], gt_bboxes[i],
                    gt_labels[i], img_metas[i])
                sampling_result = self.bbox_sampler[stage].sample(
                    assign_result, bboxes_list[i], gt_bboxes[i])
                sampling_results.append(sampling_result)
            bbox_targets = self.bbox_head[stage].get_targets(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg[stage], True)

            cls_score = bbox_results['cls_score']
            decode_bbox_pred = bbox_results['decode_bbox_pred']

            single_stage_loss = self.bbox_head[stage].loss(
                cls_score.view(-1, cls_score.size(-1)),
                decode_bbox_pred.view(-1, 4),
                *bbox_targets,
                imgs_whwh=imgs_whwh)
            for key, value in single_stage_loss.items():
                all_stage_loss[f'stage{stage}_{key}'] = value * self.stage_loss_weights[stage]

        return all_stage_loss

    def simple_test(self,
                    mlvl_feats: Tuple[torch.Tensor],
                    bboxes: torch.Tensor,
                    query_feats: torch.Tensor,
                    img_metas: List[dict],
                    imgs_whwh: torch.Tensor = None,
                    rescale: bool = False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(img_metas)
        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(stage,
                                              mlvl_feats,
                                              query_feats,
                                              bboxes,
                                              self.featmap_strides,
                                              img_metas)
            
            query_feats = bbox_results['query_feats']
            cls_score = bbox_results['cls_score']
            bboxes_list = bbox_results['detach_bboxes_list']
            bboxes = bbox_results['bboxes'].detach()

        num_classes = self.bbox_head[-1].num_classes
        det_bboxes = []
        det_labels = []

        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        for img_id in range(num_imgs):
            cls_score_per_img = cls_score[img_id]
            scores_per_img, topk_indices = cls_score_per_img.flatten(
                0, 1).topk(self.test_cfg.max_per_img, sorted=False)
            labels_per_img = topk_indices % num_classes
            bbox_pred_per_img = bboxes_list[img_id][topk_indices // num_classes]
            
            if rescale:
                scale_factor = img_metas[img_id]['scale_factor']
                bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
            det_bboxes.append(
                torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1))
            det_labels.append(labels_per_img)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], num_classes)
            for i in range(num_imgs)
        ]

        return bbox_results

    def aug_test(self, x, bboxes_list, img_metas, rescale=False):
        raise NotImplementedError()

    def forward_dummy(self, 
                      mlvl_feats: Tuple[torch.Tensor],
                      bboxes: torch.Tensor,
                      query_feats: torch.Tensor,
                      img_metas: List[dict]):
        """Dummy forward function when do the flops computing."""
        all_stage_bbox_results = []

        if self.with_bbox:
            for stage in range(self.num_stages):
                bbox_results = self._bbox_forward(stage,
                                                  mlvl_feats,
                                                  query_feats,
                                                  bboxes,
                                                  self.featmap_strides,
                                                  img_metas)
                
                all_stage_bbox_results.append(bbox_results)
                query_feats = bbox_results['query_feats']
                bboxes = bbox_results['bboxes'].detach()
        
        return all_stage_bbox_results
