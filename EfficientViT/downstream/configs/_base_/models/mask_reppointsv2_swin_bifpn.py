# model settings
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='RepPointsV2MaskDetector',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(1, 2, 3),
        use_checkpoint=False),
    neck=dict(
        type='BiFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        add_extra_convs=False,
        num_outs=5,
        no_norm_on_lateral=False,
        num_repeat=2,
        norm_cfg=norm_cfg
    ),
    bbox_head=dict(
        type='RepPointsV2Head',
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        shared_stacked_convs=1,
        first_kernel_size=3,
        kernel_size=1,
        corner_dim=64,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='RPDQualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox_init=dict(type='RPDGIoULoss', loss_weight=1.0),
        loss_bbox_refine=dict(type='RPDGIoULoss', loss_weight=2.0),
        loss_heatmap=dict(
            type='GaussianFocalLoss',
            alpha=2.0,
            gamma=4.0,
            loss_weight=0.25),
        loss_offset=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_sem=dict(
            type='SEPFocalLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.1),
        transform_method='exact_minmax',
        # new for condconv
        coord_pos='center',
        mask_head=dict(
            type='CondConvMaskHead',
            branch_cfg=dict(
                in_channels=256, # == neck out channels
                channels=128,
                in_features=[0,1,2],
                out_stride=[8,16,32], # p3, p4, p5
                norm=dict(type='BN', requires_grad=True),
                num_convs=4,
                out_channels=8,
                semantic_loss_on=False,
                num_classes=80,
                loss_sem=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0,
                    prior_prob=0.01)
            ),
            head_cfg=dict(
                channels=8,
                disable_rel_coords=False,
                num_layers=3,
                use_fp16=False,
                mask_out_stride=4,
                max_proposals=500,
                aux_loss=True,
                mask_loss_weight=[0.,0.6,1.],
                sizes_of_interest=[64, 128, 256, 512, 1024]
            ),
        )),
    train_cfg = dict(
        init=dict(
            assigner=dict(type='PointAssignerV2', scale=4, pos_num=1, mask_center_sample=True, use_center=True),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        heatmap=dict(
            assigner=dict(type='PointHMAssigner', gaussian_bump=True, gaussian_iou=0.7),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=dict(
            assigner=dict(type='ATSSAssignerV2', topk=9, mask_center_sample=True),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg = dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
)