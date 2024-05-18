_base_ = [
    '../_base_/models/danet_r50-d8.py',
    '../_base_/datasets/vaihingen.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_200e.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=6),
    auxiliary_head=dict(num_classes=6))
train_dataloader = dict(batch_size=8,
                        sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
