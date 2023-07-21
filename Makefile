fuma-r50-1x:
	CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./tools/dist_train.sh \
	configs/fuma/fuma_r50_1x_coco.py \
	2 --seed=20

fuma-r101-1x:
	CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./tools/dist_train.sh \
	configs/fuma/fuma_r101_300_query_crop_mstrain_480-800_3x_coco.py \
	2 --seed=20

fuma-r50-crop-mstrain-3x:
	CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./tools/dist_train.sh \
	configs/fuma/fuma_r50_300_query_crop_mstrain_480-800_3x_coco.py \
	2 --seed=20

fuma-r101-crop-mstrain-3x:
	CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./tools/dist_train.sh \
	configs/fuma/fuma_r101_300_query_crop_mstrain_480-800_3x_coco.py \
	2 --seed=20