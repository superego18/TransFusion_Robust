point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0] # Based on nuScenes # But, this needs to be increased to about 100.
class_names = ['car', 'truck', 'bus', 'other_vehicles', 'pedestrian', 'motorcycle', 'bicycle']
voxel_size = [0.075, 0.075, 0.2] # Based on nuScenes
out_size_factor = 8
evaluation = dict(interval=1)
dataset_type = 'RobustDataset'
data_root = '/mnt/e/Robust/transfusion' # Dataset path on external hard drive
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
img_scale = (600, 960) # Half of the original size (new height, new width)
num_views = 4 # [front, left, rear, right]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) # Traditional value (BGR)
train_pipeline = [ # Data preprocssig pipeline before training
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4), # 4 is using intensity.
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='LoadMultiViewImageFromFiles', img_scale=(1200, 1920)), # (height, width)
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.78539816, 0.78539816],
    #     scale_ratio_range=[0.95, 1.05],
    #     translation_std=[0.5, 0.5, 0.5]),
    # dict(
    #     type='RandomFlip3D',
    #     sync_2d=True,
    #     flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5),
    dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
    dict(type='MyNormalize', **img_norm_cfg),
    dict(type='MyPad', size_divisor=32), # Add black pad on right & bottom side.
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range), # Filter gt_bboxes_3d's location by bev pcd_range.
    # dict(type='ObjectNameFilter', classes=class_names), # Not required. # All gt_labels_3d are included in class_names.
    dict(type='PointShuffle'), # Shuffle points's orders in points array.
    dict(type='DefaultFormatBundle3D', class_names=class_names), # Convert data(points, img, gt_labels_3d, gt_bboxes_3d) to DataContainer of mmcv. 
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d']) # Seperate to keys(points, img, gt_labels_3d, gt_bboxes_3d) and img_metas(others). # All are DC.
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadMultiViewImageFromFiles', img_scale=(1200, 1920)), # Original img size
    dict(
    type='MultiScaleFlipAug3D', # Test time augmentation
    img_scale=img_scale,
    pts_scale_ratio=1,
    flip=False,
    transforms=[
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[0, 0],
            scale_ratio_range=[1.0, 1.0],
            translation_std=[0, 0, 0]),
        dict(type='RandomFlip3D'),
        dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
        dict(type='MyNormalize', **img_norm_cfg),
        dict(type='MyPad', size_divisor=32),
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        dict(
            type='DefaultFormatBundle3D',
            class_names=class_names,
            with_label=False),
        dict(type='Collect3D', keys=['points', 'img'])
    ])
]
data = dict(
    samples_per_gpu=1, # Support only 1 on RTX 3080. (Because of CUDA OOM in multi_head_attention_forward.)
    workers_per_gpu=6, # How many subprocesses to use for data loading
    train=dict(
        type='CBGSDataset', # Not CBGS (resampling) # This is sample augmentation.
        # type='RepeatDataset', # You can use this instead of CBGSdataset. # Simply repeat the dataset to augment.
        # times=1,  # RepeatDataset's argument
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            num_views=num_views,
            ann_file=data_root + '/robust_infos_train.pkl',
            load_interval=1, # Must set this to 1.
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        num_views=num_views,
        ann_file=data_root + '/robust_infos_val.pkl',
        load_interval=100, # You are free to change this.
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        num_views=num_views,
        ann_file=data_root + '/robust_infos_test.pkl',
        load_interval=100, # You are free to change this.
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
model = dict(
    type='TransFusionDetector',
    freeze_img=True,
    # img_backbone=dict(
    #     type='DLASeg',
    #     num_layers=34,
    #     heads={},
    #     head_convs=-1,
    #     ),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    pts_voxel_layer=dict(
        max_num_points=10, # Based on nuScenes # Maximum pts in one voxel
        voxel_size=voxel_size,
        max_voxels=(120000, 160000), # Based on nuScenes # Maximum nums of non-empty voxels (train, test)
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE', # Based on nuScenes # This simply performs the mean.
        num_features=4, # This has the same value as 'use_dim'.
    ),
    # pts_voxel_encoder=dict( # You can use this instead of HardSimpleVFE
    #     type='HardVFE',
    #     in_channels=4,
    #     feat_channels=[64],
    #     with_distance=False,
    #     with_cluster_center=False,
    #     with_voxel_center=False,
    #     voxel_size=voxel_size,
    #     norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
    #     point_cloud_range=point_cloud_range,
    # ),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4, # Number of output channels for pts_voxel_encoder
        sparse_shape=[41, 1440, 1440], # Related to (point_cloud_range / voxel_size)
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='TransFusionHead',
        fuse_img=True,
        num_views=num_views,
        in_channels_img=256,
        out_size_factor_img=4,
        num_proposals=100, # Related to maximum gt per info # In robust, max: 95, min: 0 
        auxiliary=True,
        in_channels=256 * 2,
        hidden_channel=128,
        num_classes=len(class_names),
        
        # config for Transformer
        num_decoder_layers=1,
        num_heads=8,
        learnable_query_pos=False,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        
        # config for FFN
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)), # This is maybe (output_channel, num_conv)
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0], # Range filter of 3d_bboxes's center
            score_threshold=0.0,
            code_size=8, # This is needed to encoding 3d_bboxes.
        ),
        
        # config for loss
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
        # loss_iou=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=0.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25), # Based on nuScenes
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            dataset='Robust',
            assigner=dict(
                type='HungarianAssigner3D', # Matching cost
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15), # Based on nuScenes
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25), # Based on nuScenes
                iou_cost=dict(type='IoU3DCost', weight=0.25) # Based on nuScenes
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1440, 1440, 40],  # Num of voxels
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # Based on Waymo
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='Robust',
            grid_size=[1440, 1440, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2], # Based on nuScenes
            voxel_size=voxel_size[:2],
            nms_type=None,
        )))
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)  # for 8gpu * 2sample_per_gpu
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict( # mmcv/runner/hooks/lr_updater.py/CyclicLrUpdaterHook
    policy='cyclic', # one-cycle learning rate policy # https://arxiv.org/abs/1506.01186
    target_ratio=(10, 0.0001), # Relative ratio of the highest LR and the lowest LR to the initial LR. # max_learning_rate: 0.001
    cyclic_times=1, # Number of cycles during training
    step_ratio_up=0.4) # The ratio of the increasing process of LR in the total cycle.
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1), # However un the paper, this is stated as 0.85~0.95.
    cyclic_times=1,
    step_ratio_up=0.4)
total_epochs = 6
checkpoint_config = dict(interval=1)
log_config = dict(
    interval = 50, # Cacaluate loss per 50 iters
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
# log_level = 'ERROR' # If you want to reduce log.
work_dir = None
load_from = None # Must change this to the checkpoint of TransFusion L trained on Robust.
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 8)
freeze_lidar_components = True
find_unused_parameters = True
