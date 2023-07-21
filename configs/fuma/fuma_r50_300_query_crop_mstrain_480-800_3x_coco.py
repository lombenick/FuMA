_base_ = './fuma_r50_1x_coco.py'

QUERY_DIM = 256
FF_DIM = 2048
FEAT_DIM = 256
NUM_SAMPLING_POINTS = 32
NUM_STAGES = 6
num_query = 300
model = dict(
    rpn_head=dict(num_query=num_query),
    roi_head=dict(bbox_head=[
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
            num_query=num_query,
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
    test_cfg=dict(
        _delete_=True, rpn=None, rcnn=dict(max_per_img=num_query)))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
            [
            dict(
                type='Resize',
                img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_type='absolute_range',
                crop_size=(384, 600),
                allow_negative_crop=True),
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333),
                           (576, 1333), (608, 1333), (640, 1333),
                           (672, 1333), (704, 1333), (736, 1333),
                           (768, 1333), (800, 1333)],
                multiscale_mode='value',
                override=True,
                keep_ratio=True)
        ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(samples_per_gpu=4, train=dict(pipeline=train_pipeline))
optimizer = dict(lr=1.25e-5)
lr_config = dict(policy='step', step=[24, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)