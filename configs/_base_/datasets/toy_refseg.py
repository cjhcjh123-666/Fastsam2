from mmengine.dataset import DefaultSampler
from mmdet.datasets.transforms import LoadPanopticAnnotations, RandomFlip, RandomCrop, PackDetInputs, Resize

from seg.datasets.ref_seg import RefSegDataset

data_root = 'data/'
backend_args = None
image_size = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(
        type=LoadPanopticAnnotations,
        with_bbox=True, with_mask=True, with_seg=False, backend_args=backend_args
    ),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type=PackDetInputs,
         meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=RefSegDataset,
        data_root=data_root,
        ann_file='refcoco/converted/refcoco_train.jsonl',
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)


