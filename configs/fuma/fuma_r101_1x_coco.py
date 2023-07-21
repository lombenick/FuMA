_base_ = './fuma_r50_1x_coco.py'

data = dict(samples_per_gpu=8)
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
optimizer = dict(lr=1.25e-5)