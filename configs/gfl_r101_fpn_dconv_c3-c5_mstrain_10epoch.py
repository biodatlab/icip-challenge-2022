dataset_type = 'CocoDataset'
data_root = '/workspace/data/Chula-ParasiteEgg-11/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(
                type='JpegCompression',
                quality_lower=85,
                quality_upper=95,
                p=0.2)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CocoDataset',
        ann_file=[
            '/workspace/data/Chula-ParasiteEgg-11/labels_train_with_masks.json',
            '/workspace/data/Chula-ParasiteEgg-11/labels_val_with_masks.json',
            '/workspace/data/Chula-ParasiteEgg-11/labels_test_fusion_929.json'
        ],
        img_prefix=[
            '/workspace/data/Chula-ParasiteEgg-11/train/',
            '/workspace/data/Chula-ParasiteEgg-11/val/',
            '/workspace/data/Chula-ParasiteEgg-11/test/data/'
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=[(1333, 480), (1333, 800)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Pad', size_divisor=32),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='ShiftScaleRotate',
                        shift_limit=0.0625,
                        scale_limit=0.0,
                        rotate_limit=0,
                        interpolation=1,
                        p=0.5),
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=[0.1, 0.3],
                        contrast_limit=[0.1, 0.3],
                        p=0.2),
                    dict(
                        type='JpegCompression',
                        quality_lower=85,
                        quality_upper=95,
                        p=0.2)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_labels'],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                keymap=dict(img='image', gt_bboxes='bboxes'),
                update_pad_shape=False,
                skip_img_without_anno=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=('Ascaris lumbricoides', 'Capillaria philippinensis',
                 'Enterobius vermicularis', 'Fasciolopsis buski',
                 'Hookworm egg', 'Hymenolepis diminuta', 'Hymenolepis nana',
                 'Opisthorchis viverrine', 'Paragonimus spp',
                 'Taenia spp. egg', 'Trichuris trichiura')),
    val=dict(
        type='CocoDataset',
        ann_file=
        '/workspace/data/Chula-ParasiteEgg-11/labels_val_with_masks.json',
        img_prefix='/workspace/data/Chula-ParasiteEgg-11/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('Ascaris lumbricoides', 'Capillaria philippinensis',
                 'Enterobius vermicularis', 'Fasciolopsis buski',
                 'Hookworm egg', 'Hymenolepis diminuta', 'Hymenolepis nana',
                 'Opisthorchis viverrine', 'Paragonimus spp',
                 'Taenia spp. egg', 'Trichuris trichiura')),
    test=dict(
        type='CocoDataset',
        ann_file='/workspace/data/Chula-ParasiteEgg-11/labels_test.json',
        img_prefix='/workspace/data/Chula-ParasiteEgg-11/test/data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('Ascaris lumbricoides', 'Capillaria philippinensis',
                 'Enterobius vermicularis', 'Fasciolopsis buski',
                 'Hookworm egg', 'Hymenolepis diminuta', 'Hymenolepis nana',
                 'Opisthorchis viverrine', 'Paragonimus spp',
                 'Taenia spp. egg', 'Trichuris trichiura')))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7, 9])
runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'pretrained_models/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20200630_102002-134b07df.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
model = dict(
    type='GFL',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFLHead',
        num_classes=11,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2)
]
classes = ('Ascaris lumbricoides', 'Capillaria philippinensis',
           'Enterobius vermicularis', 'Fasciolopsis buski', 'Hookworm egg',
           'Hymenolepis diminuta', 'Hymenolepis nana',
           'Opisthorchis viverrine', 'Paragonimus spp', 'Taenia spp. egg',
           'Trichuris trichiura')
fp16 = dict(loss_scale=512.0)
work_dir = './work_dirs/gfl_r101_fpn_dconv_c3-c5_mstrain_10e_chulacell'
auto_resume = False
gpu_ids = [0]
