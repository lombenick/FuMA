log_interval = 100

_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
work_dir_prefix = 'work_dirs/FuMA_workdir'

NUM_STAGES = 6
NUM_QUERY = 100
QUERY_DIM = 256
FEAT_DIM = 256
FF_DIM = 2048
NUM_SAMPLING_POINTS = 32

data = dict(samples_per_gpu=4, workers_per_gpu=2)

model = dict(
    type='FuMA',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='ChannelMapping',              # 调用class ChannelMapping 不同通道数的多尺度特征图 -> 相同通道数的特征图
        in_channels=[256, 512, 1024, 2048], # 输入的特征图通道数目
        out_channels=FEAT_DIM,              # 输出的通道数目
        start_level=0,                      # 从0号特征图开始
        add_extra_convs='on_output',        # 额外的特征图的来源
        num_outs=4),                        # 输出4张特征图
    rpn_head=dict(
        type='InitQueryHead',
        num_query=NUM_QUERY,
        content_dim=QUERY_DIM),
    roi_head=dict(
        type='FuMADecoder',
        num_stages=NUM_STAGES,
        stage_loss_weights=[1] * NUM_STAGES,
        query_dim=QUERY_DIM,
        featmap_strides=[4, 8, 16, 32],
        bbox_head=[
            dict(
                type='FuMADecoderStage',
                num_classes=80,
                num_ffn_fcs=2,
                num_cls_fcs=1,
                num_reg_fcs=1,
                feedforward_channels=FF_DIM,
                query_dim=QUERY_DIM,
                feats_dim=FEAT_DIM,
                dropout=0.0,
                ffn_act_cfg=dict(type='GELU'),
                num_levels=4,
                num_sampling_points=NUM_SAMPLING_POINTS,
                num_query=NUM_QUERY,
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                # NOTE: The following argument is a placeholder to hack the code. No real effects for decoding or updating bounding boxes.
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(NUM_STAGES)
        ]),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                    iou_cost=dict(type='IoUCost', iou_mode='giou',
                                  weight=2.0)),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1) for _ in range(NUM_STAGES)
        ]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=NUM_QUERY)))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1.25e-5,
    weight_decay=0.0001,
)

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1.0, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    step=[8, 11],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001
)


def __date():
    import datetime
    return datetime.datetime.now().strftime('%m%d_%H%M')


log_config = dict(
    interval=log_interval,
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)

postfix = '_' + __date()
find_unused_parameters = True
resume_from = None