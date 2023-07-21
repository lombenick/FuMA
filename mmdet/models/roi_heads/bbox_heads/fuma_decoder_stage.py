import math
import mmcv
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Optional, Tuple
from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import FFN
from mmcv.runner import auto_fp16, force_fp32
from mmdet.core import multi_apply, bbox_overlaps
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
from .bbox_head import BBoxHead


def decode_box(bboxes):
    '''decode the bboxes from (xc, yc, scale, ratio) to (x1, y1, x2, y2)'''
    scale = 2.00 ** bboxes[..., 2:3]
    ratio = 2.00 ** torch.cat([bboxes[..., 3:4] * -0.5,
                              bboxes[..., 3:4] * 0.5], dim=-1)
    wh = scale * ratio
    xy = bboxes[..., 0:2]
    roi = torch.cat([xy - wh * 0.5, xy + wh * 0.5], dim=-1)
    return roi

def position_embedding_for_queries(bboxes: torch.Tensor, 
                                   num_feats: int = 64, 
                                   temperature: int = 10000):
    """Get the position embeddings of proposals."""
    term = bboxes.new_tensor([1000, 1000, 1, 1]).view(1, 1, -1)
    bboxes = bboxes / term
    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=bboxes.device)
    dim_t = (temperature ** (dim_t / num_feats)).view(1, 1, 1, -1)
    pos_x = bboxes[..., None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),dim=4).flatten(2)
    return pos_x

def sampling_each_level(sample_points: torch.Tensor,
                        value: torch.Tensor,
                        weight: Optional[torch.Tensor] = None,
                        n_points: int = 32, 
                        num_heads: int = 4, 
                        channel_dim: int = 64):
    B, num_query = sample_points.size()[:2]
    H, W = value.size()[2:]

    # `sampling_points` should reshape with (B*num_heads, num_query, n_points, 2)
    sample_points = sample_points \
        .view(B, num_query, num_heads, n_points, 2) \
        .permute(0, 2, 1, 3, 4).flatten(0, 1)
    sample_points = 2.0 * sample_points - 1.0

    # `out`` should reshape (B*num_heads, C, num_query, n_points)
    value = value.view(B * num_heads, channel_dim, H, W)
    out = F.grid_sample(value, sample_points,
            mode='bilinear', padding_mode='zeros', align_corners=False)

    # `weight`` should reshape (B*num_heads, 1, num_query, n_points)
    if weight is not None:
        weight = weight.view(B, num_query, num_heads, n_points) \
            .permute(0, 2, 1, 3).flatten(0, 1).unsqueeze(1)
        out *= weight

    # `out`` should reshape (B, num_query, num_heads, n_points, channels_dim)
    return out \
        .view(B, num_heads, channel_dim, num_query, n_points) \
        .permute(0, 3, 1, 4, 2)

def get_sampling_features(sampled_xy: torch.Tensor,
                          mlvl_feats: Tuple[torch.Tensor],
                          level_weights: torch.Tensor,
                          featmap_strides: List[int] = [4, 8, 16, 32],
                          n_points: int = 32,
                          num_heads: int = 4,
                          num_levels: int = 4):
    '''
        args:
            normalized_xy: tensor, has shape (B, num_query, _t, n_groups_points, 2)(real coordinate)
            level_weights: Tensor, has shape (B, num_query, n_groups * n_points * num_levels)

        returns:
            out: tensor, shape of (B, num_query, n_groups, n_points, channels_dim) 
    '''
    B, num_query, _, num_heads_points = sampled_xy.size()[:4]
    B, C = mlvl_feats[0].size()[:2]
    channel_dim = C // num_heads

    level_weights = level_weights.view(B, num_query, num_heads_points, -1)
    sample_points_lvl_weight_list = level_weights.unbind(-1)

    out = sampled_xy.new_zeros(B, num_query, num_heads, n_points, channel_dim)

    for i in range(num_levels):
        value = mlvl_feats[i]
        lvl_weights = sample_points_lvl_weight_list[i]
        mapping_size = value.new_tensor(
            [value.size(3), value.size(2)]).view(1, 1, 1, 1, -1) * featmap_strides[i]
        normalized_xy = sampled_xy / mapping_size
        out += sampling_each_level(normalized_xy, value, weight=lvl_weights, n_points=n_points,
            num_heads=num_heads, channel_dim=channel_dim)

    return out

def get_sample_points(offset: torch.Tensor, 
                      num_heads_points: int, 
                      bboxes: torch.Tensor):
    '''
        args:
            offset (Tensor): with shape (B, L, num_group_points*2), normalized by stride
            num_group_points (int): The number of the sampling groups * the num of sampling points. 
            bbox: with shape (B, L, 4) (real coordinate)

        return:
            [B, L, 1, num_group_points, 2]
        '''
    B, num_query, _ = offset.size()
    offset = offset.view(B, num_query, 1, num_heads_points, 2)
    scale = 2.00 ** bboxes[..., 2:3]
    ratio = 2.00 ** torch.cat([bboxes[..., 3:4] * -0.5,
        bboxes[..., 3:4] * 0.5], dim=-1)
    roi_wh = scale * ratio
    roi_cc = bboxes[..., :2]
    scale = scale[..., None].repeat(1, 1, num_heads_points, 1)

    # y与h相对, x与w相对
    offset_xy = offset * roi_wh.view(B, num_query, 1, 1, 2)
    sample_xy = roi_cc.contiguous().view(B, num_query, 1, 1, 2) + offset_xy

    return sample_xy, scale


class FeatureMixer(nn.Module):
    def __init__(self, 
                 feats_dim: int = 256, 
                 num_points: int = 32, 
                 num_heads: int = 4, 
                 query_dim: int = 256,
                 expansion_factor: int = 2):
        super(FeatureMixer, self).__init__()
        expansion_num_points = expansion_factor * num_points
        self.param1 = feats_dim * expansion_num_points
        self.param2 = expansion_num_points * num_points * num_heads
        self.total_parameters = self.param1 + self.param2
        self.parameter_generator = nn.Linear(query_dim, self.total_parameters)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(self.param2 * expansion_factor, query_dim)
        self.norm = nn.LayerNorm(query_dim)

        self.init_weights()
    
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def forward(self, 
                query_feats: torch.Tensor, 
                sampled_features: torch.Tensor):
        B, num_query, num_heads, num_points, C = sampled_features.shape

        '''generate mixing parameters'''
        params = self.parameter_generator(query_feats)
        params = params.reshape(B * num_query, -1)
        param1, param2 = params.split([self.param1, self.param2], 1)
        param1 = param1.reshape(B, num_query, num_heads, C, -1)
        param2 = param2.reshape(B, num_query, num_heads, -1, num_points)
        out = sampled_features
        
        '''mapping to (E * Nq, Nq) and mixing'''
        out = torch.matmul(out, param1)
        out = self.act(out)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])

        '''spatial mixing, expand to (E * Nq, E * Nq)'''
        out = torch.matmul(param2, out)
        out = self.act(out)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        
        '''back (1, Nh * E * Nq * Nq) to query dim'''
        out = out.reshape(B, num_query, -1)
        out = self.out_proj(out)
        out = self.norm(out)

        return out


class FeatureMixerHead(nn.Module):
    def __init__(self,
                 num_points: int = 32,
                 num_heads: int = 4,
                 query_dim: int = 256,
                 num_levels: int = 4,
                 feats_dim: int = 256):
        super(FeatureMixerHead, self).__init__()
        self.num_points = num_points
        self.num_heads = num_heads
        self.sampling_offset_generator = nn.Linear(query_dim, num_points * num_heads * 2)
        self.level_weights_generator = nn.Sequential(nn.Linear(1, num_levels), nn.Softmax(dim=-1))
        self.mlp_mixer = FeatureMixer(feats_dim=feats_dim, query_dim=query_dim,
            num_points=num_points, num_heads=num_heads)
        self.norm = nn.LayerNorm(query_dim)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.sampling_offset_generator.weight)
        nn.init.zeros_(self.sampling_offset_generator.bias)
        nn.init.zeros_(self.level_weights_generator[0].weight)
        nn.init.uniform_(self.level_weights_generator[0].bias, a=0, b=10)
        bias = self.sampling_offset_generator.bias.data.view(
            self.num_heads, self.num_points, 2)
        nn.init.uniform_(bias, -0.5, 0.5)
        self.mlp_mixer.init_weights()

    def forward(self, 
                mlvl_feats: Tuple[torch.Tensor], 
                query_feats: torch.Tensor, 
                bboxes: torch.Tensor, 
                featmap_strides: List[int]):
        '''
            args:
                bbox: tensor, has shape (B, num_query, 4)(real coordinate)
                level_weights: Tensor, has shape (B, num_query, 4)
        '''
        offset = self.sampling_offset_generator(query_feats)
        sampled_xy, scale = get_sample_points(
            offset, self.num_heads * self.num_points, bboxes)
        level_weights = self.level_weights_generator(scale)
        
        sampled_feature = get_sampling_features(sampled_xy, mlvl_feats,
            level_weights=level_weights, featmap_strides=featmap_strides,
            n_points=self.num_points, num_heads=self.num_heads)
        query_feats = query_feats + self.mlp_mixer(query_feats, sampled_feature)
        query_feats = self.norm(query_feats)

        return query_feats
    

class QueryMixer(nn.Module):
    def __init__(self, query_dim: int = 256, num_query: int = 100, 
        num_heads: int = 8, expansion_factor: int = 4):
        super(QueryMixer, self).__init__()
        self.num_heads = num_heads
        self.param1 = query_dim
        self.weight_bank = nn.Parameter(torch.ones(1, num_heads, num_query, num_query)
                                        / math.sqrt(query_dim // num_heads))
        self.params_generator = nn.Linear(query_dim, self.param1)
        self.trans_value = nn.Linear(query_dim, query_dim * expansion_factor)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(query_dim * expansion_factor, query_dim)
        # self.norm = nn.LayerNorm(query_dim)

        self.init_weight()
    
    @torch.no_grad()
    def init_weight(self):
        nn.init.zeros_(self.params_generator.weight)

    def forward(self, query_feats: torch.Tensor, bias: torch.Tensor):
        bs, num_query, _ = query_feats.size()
        
        M = self.params_generator(query_feats)
        M = M.reshape(bs, num_query, self.num_heads, -1).permute(0, 2, 3, 1)
        value = self.trans_value(query_feats).reshape(bs, num_query, 
            self.num_heads, -1).transpose(1, 2)
        out = query_feats.reshape(bs, num_query, self.num_heads, -1).transpose(1, 2)

        out = torch.matmul(out, M) * self.weight_bank + bias
        out = self.act(out)
        out = F.softmax(out, -1)

        out = torch.matmul(out, value)
        out = out.transpose(1, 2).reshape(bs, num_query, -1)
        out = self.out_proj(out)
        # out = self.norm(out)
        out = out + query_feats

        return out

@HEADS.register_module()
class FuMADecoderStage(BBoxHead):
    r"""this is used for deaod_encoder

    Args:
        num_classes (int): Number of class in dataset.
            Defaults to 80.
        num_ffn_fcs (int): The number of fully-connected
            layers in FFNs. Defaults to 2.
        num_heads (int): The hidden dimension of FFNs.
            Defaults to 8.
        num_groups (int): The num of groups which used in
            mixer head.Defaults to 4.
        num_cls_fcs (int): The number of fully-connected
            layers in classification subnet. Defaults to 1.
        num_reg_fcs (int): The number of fully-connected
            layers in regression subnet. Defaults to 1.
        feedforward_channels (int): The hidden dimension
            of FFNs. Defaults to 2048.
        query_dim (int): The embedding dims of the query.
            Defaults to 256.
        feats_dim (int): The channels dim of the feature
            maps.Defaults to 256.
        dropout (float): Probability of drop the channel.
            Defaults to 0.0.
        ffn_act_cfg (dict): The activation config for FFNs.
        num_levels (int): The number of feature maps.Defaults
            to 4.
        num_sampling_points (int): The number of
            sampling points which will be used in mixer head.
            Defaults to 32.
        loss_iou (dict): The config for iou or giou loss.

    """

    def __init__(self,
                 num_classes: int = 80,
                 num_ffn_fcs: int = 2,
                 num_query_mixer_heads: int = 8,
                 num_feature_mixer_heads: int = 4,
                 num_cls_fcs: int = 1,
                 num_reg_fcs: int = 1,
                 feedforward_channels: int = 2048,
                 query_dim: int = 256,
                 feats_dim: int = 256,
                 dropout: float = 0.0,
                 ffn_act_cfg: Optional[mmcv.ConfigDict] = dict(type='GELU'),
                 num_levels: int = 4,
                 num_sampling_points: int = 32,
                 num_query: int = 100,
                 loss_iou: Optional[mmcv.ConfigDict] = dict(type='GIoULoss', loss_weight=2.0),
                 init_cfg: Optional[mmcv.ConfigDict] = None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(FuMADecoderStage, self).__init__(
            num_classes=num_classes,
            reg_decoded_bbox=True,
            reg_class_agnostic=True,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_iou = build_loss(loss_iou)
        self.query_dim = query_dim
        self.num_levels = num_levels
        self.num_query = num_query
        self.fp16_enabled = False

        self.query_mixer = QueryMixer(query_dim, num_query, num_query_mixer_heads)
        self.mixer_norm = build_norm_layer(dict(type='LN'), query_dim)[1]

        self.ffn = FFN(query_dim, feedforward_channels, num_ffn_fcs, 
            act_cfg=ffn_act_cfg, dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), query_dim)[1]

        # 分类全连接层的定义
        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(nn.Linear(query_dim, query_dim, bias=True))
            self.cls_fcs.append(build_norm_layer(dict(type='LN'), query_dim)[1])
            self.cls_fcs.append(build_activation_layer(dict(type='ReLU', inplace=True)))

        # overload the self.fc_cls in BBoxHead
        # fully connected layers for classification.The sigmoid method is used by default.
        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(query_dim, self.num_classes)
        else:
            self.fc_cls = nn.Linear(query_dim, self.num_classes + 1)

        # 回归全连接层的定义
        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(nn.Linear(query_dim, query_dim, bias=True))
            self.reg_fcs.append(build_norm_layer(dict(type='LN'), query_dim)[1])
            self.reg_fcs.append(build_activation_layer(dict(type='ReLU', inplace=True)))
        # overload the self.fc_cls in BBoxHead
        # Regress the next bbox position for updating.fully connected layers for regression
        self.fc_reg = nn.Linear(query_dim, 4)

        # mixer head
        self.feature_mixer = FeatureMixerHead(
            num_points=num_sampling_points,
            num_heads=num_feature_mixer_heads,
            query_dim=query_dim,
            num_levels=num_levels,
            feats_dim=feats_dim)
        
        self.iof_tau = nn.Parameter(torch.ones(num_query_mixer_heads, ))

    @torch.no_grad()
    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        super(FuMADecoderStage, self).init_weights()
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
                nn.init.xavier_uniform_(m.weight)

        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)

        nn.init.zeros_(self.fc_reg.weight)
        nn.init.zeros_(self.fc_reg.bias)

        nn.init.uniform_(self.iof_tau, 0.0, 4.0)

        self.feature_mixer.init_weights()
        # self.query_mixer.init_weight()

    @auto_fp16()
    def forward(self, 
                mlvl_feats: Tuple[torch.Tensor],
                query_feats: torch.Tensor, 
                bboxes: torch.Tensor,
                featmap_strides: List[int]):
        """Forward function of Deaod Encoder Stage Head.

        Args:
            mlvl_feats (Tuple[Tensor]): the feature maps of the images, has 
                shape (num_levels, batch_size, embed_dims, H, W)(Make the H 
                and W of the feature map in a batch size the same after padding)
            query_feats (Tensor): query features with shape (batch_size, num_query,
                embed_dims).
            bbox (Tensor): shape (batch_size, num_query, 4) (real coordinate)
            featmap_strides (list): like [4, 8, 16, 32]

          Returns:
                tuple[Tensor]: Usually a tuple of classification scores
                and bbox prediction and a intermediate feature.

                    - cls_scores (Tensor): Classification scores for
                        all proposals, has shape (batch_size, num_query, 
                        num_classes).
                    - bbox_preds (Tensor): Box energies / deltas for
                        all proposals, has shape (batch_size, num_query, 4).
                    - query_feats (Tensor): Object feature before classification
                        and regression subnet, has shape (batch_size, num_query, 
                        embed_dims).
        """
        batch_size, num_query, _ = query_feats.size()
        with torch.no_grad():
            rois = decode_box(bboxes)
            roi_box_batched = rois.view(batch_size, num_query, 4)
            iof = bbox_overlaps(roi_box_batched, roi_box_batched, mode='iof')[
                :, None, :, :]
            iof = (iof + 1e-7).log()
            pe = position_embedding_for_queries(bboxes, query_feats.size(-1) // 4)
        bias = (iof * self.iof_tau.view(1, -1, 1, 1))
        query_feats = query_feats + pe

        '''query mixing'''
        query_feats = self.query_mixer(query_feats, bias)
        query_feats = self.mixer_norm(query_feats)

        '''mixer head + residual + LN'''
        query_feats = self.feature_mixer(
            mlvl_feats, query_feats, bboxes, featmap_strides)
        
        '''2 * FFN (linear + residual + norm)'''
        query_feats = self.ffn_norm(self.ffn(query_feats))

        cls_feat = query_feats
        reg_feat = query_feats

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)

        cls_score = self.fc_cls(cls_feat).view(batch_size, num_query, -1)
        bbox_delta = self.fc_reg(reg_feat).view(batch_size, num_query, -1)

        return cls_score, bbox_delta, query_feats.view(batch_size, num_query, -1)

    def refine_xysr(self, bbox, bbox_delta):
        scale = 2 ** bbox[..., 2:3]
        new_xy = bbox[..., 0:2] + bbox_delta[..., 0:2] * scale
        new_scale_ratio = bbox[..., 2:4] + bbox_delta[..., 2:4]
        bbox = torch.cat([new_xy, new_scale_ratio], dim=-1)
        return bbox, decode_box(bbox)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score: torch.Tensor,
             bbox_pred: torch.Tensor,
             labels: torch.Tensor,
             label_weights: torch.Tensor,
             bbox_targets: torch.Tensor,
             bbox_weights: torch.Tensor,
             imgs_whwh: Optional[torch.Tensor] = None,
             reduction_override: Optional[str] = None,
             **kwargs):
        """"Loss function of Deaod Encoder Stage Head, get loss of all images.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            labels (Tensor): Label of each proposals, has shape
                (batch_size * num_proposals_single_image
            label_weights (Tensor): Classification loss
                weight of each proposals, has shape
                (batch_size * num_proposals_single_image
            bbox_targets (Tensor): Regression targets of each
                proposals, has shape
                (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents
                [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression loss weight of each
                proposals's coordinate, has shape
                (batch_size * num_proposals_single_image, 4),
            imgs_whwh (Tensor): imgs_whwh (Tensor): Tensor with
                shape (batch_size, num_proposals, 4), the last
                dimension means
                [img_width,img_height, img_width, img_height].
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

            Returns:
                dict[str, Tensor]: Dictionary of loss components
        """
        losses = dict()
        bg_class_ind = self.num_classes

        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos)
        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['pos_acc'] = accuracy(cls_score[pos_inds],
                                             labels[pos_inds])
        if bbox_pred is not None:
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                pos_bbox_pred = bbox_pred.reshape(bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                imgs_whwh = imgs_whwh.reshape(bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred / imgs_whwh,
                    bbox_targets[pos_inds.type(torch.bool)] / imgs_whwh,
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
                losses['loss_iou'] = self.loss_iou(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
                losses['loss_iou'] = bbox_pred.sum() * 0
        return losses

    def _get_target_single(self, pos_inds: torch.Tensor, neg_inds: torch.Tensor, 
                           pos_bboxes: torch.Tensor, neg_bboxes: torch.Tensor,
                           pos_gt_bboxes: torch.Tensor, pos_gt_labels: torch.Tensor,
                           cfg: Optional[mmcv.ConfigDict]):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Almost the same as the implementation in `bbox_head`,
        we add pos_inds and neg_inds to select positive and
        negative samples instead of selecting the first num_pos
        as positive samples.

        Args:
            pos_inds (Tensor): The length is equal to the
                positive sample numbers contain all index
                of the positive sample in the origin proposal set.
            neg_inds (Tensor): The length is equal to the
                negative sample numbers contain all index
                of the negative sample in the origin proposal set.
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains all the gt_boxes,
                has shape (num_gt, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains all the gt_labels,
                has shape (num_gt).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all proposals, has
                  shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all proposals, has
                  shape (num_proposals, 4), the last dimension 4
                  represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all proposals,
                  has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1
        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results: List[object],
                    gt_bboxes: List[torch.Tensor],
                    gt_labels: List[torch.Tensor],
                    rcnn_train_cfg: Optional[mmcv.ConfigDict],
                    concat: bool = True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:`ConfigDict`): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise just
                  a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals,) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list has
                  shape (num_proposals, 4) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals, 4),
                  the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights