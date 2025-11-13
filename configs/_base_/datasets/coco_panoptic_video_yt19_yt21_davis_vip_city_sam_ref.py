# dataset settings
from mmengine import read_base
from mmengine.dataset import DefaultSampler, RepeatDataset

from seg.datasets.concat_dataset import ConcatOVDataset
from seg.datasets.samplers.batch_sampler import VideoSegAspectRatioBatchSampler

# 导入所有数据集的train_dataloader配置
with read_base():
    from .coco_panoptic_video_lsj import train_dataloader as _coco_vid_train_loader
    from .youtube_vis_2019 import train_dataloader as _yt19_train_loader
    from .youtube_vis_2021 import train_dataloader as _yt21_train_loader
    from .davis import train_dataloader as _davis_train_loader
    from .vipseg import train_dataloader as _vip_train_loader
    from .cityscapes_panoptic_720p import train_dataloader as _city_train_loader

from mmcv.transforms import LoadImageFromFile, RandomResize
from mmengine.dataset import DefaultSampler

from mmdet.datasets.transforms import LoadPanopticAnnotations, LoadAnnotations, RandomFlip, RandomCrop, PackDetInputs, Resize

from seg.datasets.coco_ov import CocoPanopticOVDataset
from seg.datasets.ref_seg import RefSegDataset
from seg.datasets.pipelines.loading import FilterAnnotationsHB
from seg.datasets.pipelines.formatting import GeneratePoint
from seg.datasets.pipelines.loading import LoadPanopticAnnotationsAll

data_root = 'data/coco/'
backend_args = None
image_size = (1280, 736)

num_mask_tokens = 1

# SAM数据集配置 (用于类别无关的分割任务)
sam_train_pipeline = [
    dict(type=LoadImageFromFile, to_float32=True, backend_args=backend_args),
    dict(type=LoadPanopticAnnotationsAll),
    dict(type=RandomFlip, prob=0.5),
    dict(type=RandomResize, resize_type=Resize, scale=image_size, ratio_range=(0.1, 2.0), keep_ratio=True),
    dict(type=RandomCrop, crop_size=image_size, crop_type='absolute', recompute_bbox=True, allow_negative_crop=True),
    dict(type=FilterAnnotationsHB, by_box=False, by_mask=True, min_gt_mask_area=32),
    dict(type=PackDetInputs),
    dict(type=GeneratePoint, num_proposals=30, num_mask_tokens=num_mask_tokens)
]

sam_dataset = dict(
    type=CocoPanopticOVDataset,
    data_root=data_root,
    ann_file='annotations/panoptic_train2017.json',
    data_prefix=dict(img='train2017/', seg='annotations/panoptic_train2017/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=sam_train_pipeline,
    backend_args=backend_args
)

# RefCOCO数据集配置 (文本交互分割任务)
# 注意：RefCOCO数据集使用COCO 2014图像，但代码会自动转换为COCO 2017格式
# 如果您的COCO图像在其他位置，请修改data_prefix路径
ref_seg_data_root = 'data/ref_seg/'
refcoco_dataset = dict(
    type=RefSegDataset,
    data_root=ref_seg_data_root,
    ann_file='refcoco',  # 数据集目录名
    dataset_name='refcoco',  # 使用 REFER API
    split_by='unc',  # refcoco 使用 'unc' split
    split='train',  # 训练集
    max_sentences_per_image=3,  # 每个图像最多采样3个句子
    debug=False,
    # 使用 COCO 2017 图像（代码会自动将 COCO 2014 文件名转换为 COCO 2017 格式）
    # 注意：如果使用相对路径，请确保路径相对于项目根目录
    # 如果需要使用绝对路径，请修改为绝对路径，例如: '/mnt/chenjiahui/Fastsam2-main/data/coco/train2017/'
    data_prefix=dict(img='../coco/train2017/'),  # 相对路径，相对于项目根目录
    pipeline=[
        dict(type=LoadImageFromFile, to_float32=True, backend_args=backend_args),
        dict(
            type=LoadAnnotations,  # 使用标准的 LoadAnnotations，能解析 RLE mask
            with_bbox=True, with_mask=True, with_seg=False, backend_args=backend_args
        ),
        dict(type=RandomFlip, prob=0.5),
        dict(type=RandomResize, resize_type=Resize, scale=image_size, ratio_range=(0.1, 2.0), keep_ratio=True),
        dict(
            type=RandomCrop,
            crop_size=image_size,
            crop_type='absolute',
            recompute_bbox=True,
            allow_negative_crop=True
        ),
        # 过滤掉裁剪后无效的实例（mask 面积太小或为空）
        dict(type=FilterAnnotationsHB, by_box=False, by_mask=True, min_gt_mask_area=32),
        dict(type=PackDetInputs,
             meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','text')),
        # generate point prompts from gt masks to align with current prompt pipeline
        dict(type=GeneratePoint, num_proposals=30, num_mask_tokens=num_mask_tokens)
    ],
    backend_args=backend_args
)

# 联合数据集配置
# 包含所有8个数据集: coco, yt19, yt21, davis, vip, city, sam, refcoco
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=VideoSegAspectRatioBatchSampler),
    dataset=dict(
        type=ConcatOVDataset,
        # data_tag与datasets列表顺序对应，用于标识数据来源
        data_tag=('coco', 'yt19', 'yt21', 'davis', 'vip', 'city', 'sam', 'refcoco'),
        datasets=[
            # 1. COCO Panoptic Video数据集
            dict(
                type=RepeatDataset,
                dataset=_coco_vid_train_loader.dataset,
                times=1,
            ),
            # 2. YouTube VIS 2019数据集
            dict(
                type=RepeatDataset,
                dataset=_yt19_train_loader.dataset,
                times=25,
            ),
            # 3. YouTube VIS 2021数据集
            dict(
                type=RepeatDataset,
                dataset=_yt21_train_loader.dataset,
                times=25,
            ),
            # 4. DAVIS数据集
            dict(
                type=RepeatDataset,
                dataset=_davis_train_loader.dataset,
                times=10,
            ),
            # 5. VIPSeg数据集
            dict(
                type=RepeatDataset,
                dataset=_vip_train_loader.dataset,
                times=10,
            ),
            # 6. Cityscapes数据集
            dict(
                type=RepeatDataset,
                dataset=_city_train_loader.dataset,
                times=5,
            ),
            # 7. SAM数据集 (类别无关分割)
            dict(
                type=RepeatDataset,
                dataset=sam_dataset,
                times=1,
            ),
            # 8. RefCOCO数据集 (文本交互分割)
            dict(
                type=RepeatDataset,
                dataset=refcoco_dataset,
                times=5,
            ),
        ]
    ),
)