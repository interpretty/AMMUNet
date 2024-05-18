_base_ = [
    '../_base_/models/ocrnet_hr18.py',
    '../_base_/datasets/vaihingen.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_200e.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://msra/hrnetv2_w18_small',
    backbone=dict(
        extra=dict(
            stage1=dict(num_blocks=(2, )),
            stage2=dict(num_blocks=(2, 2)),
            stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
            stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[18, 36, 72, 144],
            channels=sum([18, 36, 72, 144]),
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=6,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[18, 36, 72, 144],
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            channels=512,
            ocr_channels=256,
            dropout_ratio=-1,
            num_classes=6,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ])
train_dataloader = dict(batch_size=8,
                        sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
