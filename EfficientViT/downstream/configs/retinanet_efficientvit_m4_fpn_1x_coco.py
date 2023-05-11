_base_ = [
    './_base_/models/retinanet_efficientvit_fpn.py',
    './_base_/datasets/coco_detection.py',
    './_base_/schedules/schedule_1x.py', 
    './_base_/default_runtime.py'
]

model = dict(
    pretrained=None,
    backbone=dict(        
        type='EfficientViT_M4',
        pretrained="/root/efficientvit_m4.pth",
        frozen_stages=-1,
        ),
    neck=dict(
        type='EfficientViTFPN',
        in_channels=[128, 256, 384],
        out_channels=256,
        start_level=0,
        num_outs=5,
        num_extra_trans_convs=1,
        ))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'attention_biases': dict(decay_mult=0.),
                                                 'attention_bias_idxs': dict(decay_mult=0.),
                                                 }))
# optimizer_config = dict(grad_clip=None)
# do not use mmdet version fp16
# fp16 = None
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12
