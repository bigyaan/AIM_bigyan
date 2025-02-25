model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ViT_CLIP',
        input_resolution=224,
        patch_size=16,
        num_frames=8,
        width=768,
        layers=12,
        heads=12,
        drop_path_rate=0.2,
        adapter_scale=1,
        num_tadapter=2,
        pretrained='openaiclip'),
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        num_classes=174,
        spatial_type='avg',
        dropout_ratio=0.5),
    test_cfg=dict(average_clips='prob', max_testing_views=2),
    train_cfg=dict(
        blending=dict(type='LabelSmoothing', num_classes=174, smoothing=0.1)))
checkpoint_config = dict(interval=10)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
dataset_type = 'VideoDataset'
data_root = 'data/data/sthv2/test'
data_root_val = 'data/data/sthv2/test'
ann_file_train = 'data/data/sthv2/test_video_list.txt'
ann_file_val = 'data/data/sthv2/test_video_list.txt'
ann_file_test = 'data/data/sthv2/test_video_list.txt'
img_norm_cfg = dict(
    mean=[122.769, 116.74, 104.04], std=[68.493, 66.63, 70.321], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=2,
        num_clips=1,
        frame_uniform=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0),
    dict(type='Imgaug', transforms=[dict(type='RandAugment', n=4, m=7)]),
    dict(
        type='Normalize',
        mean=[122.769, 116.74, 104.04],
        std=[68.493, 66.63, 70.321],
        to_bgr=False),
    dict(type='RandomErasing', probability=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=2,
        num_clips=1,
        frame_uniform=True,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(
        type='Normalize',
        mean=[122.769, 116.74, 104.04],
        std=[68.493, 66.63, 70.321],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=2,
        num_clips=1,
        frame_uniform=True,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(
        type='Normalize',
        mean=[122.769, 116.74, 104.04],
        std=[68.493, 66.63, 70.321],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=1,
    workers_per_gpu=1,
    val_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
    train=dict(
        type='VideoDataset',
        ann_file='data/data/sthv2/test_video_list.txt',
        data_prefix='data/data/sthv2/test',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=2,
                num_clips=1,
                frame_uniform=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='RandomResizedCrop'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0),
            dict(
                type='Imgaug', transforms=[dict(type='RandAugment', n=4,
                                                m=7)]),
            dict(
                type='Normalize',
                mean=[122.769, 116.74, 104.04],
                std=[68.493, 66.63, 70.321],
                to_bgr=False),
            dict(type='RandomErasing', probability=0.25),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    val=dict(
        type='VideoDataset',
        ann_file='data/data/sthv2/test_video_list.txt',
        data_prefix='data/data/sthv2/test',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=2,
                num_clips=1,
                frame_uniform=True,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Flip', flip_ratio=0),
            dict(
                type='Normalize',
                mean=[122.769, 116.74, 104.04],
                std=[68.493, 66.63, 70.321],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]),
    test=dict(
        type='VideoDataset',
        ann_file='data/data/sthv2/test_video_list.txt',
        data_prefix='data/data/sthv2/test',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=2,
                num_clips=1,
                frame_uniform=True,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 224)),
            dict(type='ThreeCrop', crop_size=224),
            dict(type='Flip', flip_ratio=0),
            dict(
                type='Normalize',
                mean=[122.769, 116.74, 104.04],
                std=[68.493, 66.63, 70.321],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
optimizer = dict(
    type='AdamW',
    lr=0.0003,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            class_embedding=dict(decay_mult=0.0),
            positional_embedding=dict(decay_mult=0.0),
            ln_1=dict(decay_mult=0.0),
            ln_2=dict(decay_mult=0.0),
            ln_pre=dict(decay_mult=0.0),
            ln_post=dict(decay_mult=0.0))))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5)
total_epochs = 50
work_dir = 'work_dirs_vit_triplet'
find_unused_parameters = False
gpu_ids = range(0, 1)
omnisource = False
module_hooks = []
