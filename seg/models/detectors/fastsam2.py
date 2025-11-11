
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


from mmdet.models.detectors import SingleStageDetector
from .mask2former_vid import Mask2formerVideo

@MODELS.register_module()
class Fastsam2(Mask2formerVideo):
    OVERLAPPING = None

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 panoptic_head: OptConfigType = None,
                 panoptic_fusion_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 inference_sam: bool = False,
                 init_cfg: OptMultiConfig = None
                 ):
        super(SingleStageDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        panoptic_head_ = panoptic_head.deepcopy()
        panoptic_head_.update(train_cfg=train_cfg)
        panoptic_head_.update(test_cfg=test_cfg)
        self.panoptic_head = MODELS.build(panoptic_head_)

        panoptic_fusion_head_ = panoptic_fusion_head.deepcopy()
        panoptic_fusion_head_.update(test_cfg=test_cfg)
        self.panoptic_fusion_head = MODELS.build(panoptic_fusion_head_)

        self.num_things_classes = self.panoptic_head.num_things_classes
        self.num_stuff_classes = self.panoptic_head.num_stuff_classes
        self.num_classes = self.panoptic_head.num_classes

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.alpha = 0.4
        self.beta = 0.8

        self.inference_sam = inference_sam

    # -------- dynamic routing (path gating) --------
    def _apply_routing(self, batch_inputs, batch_data_samples):
        try:
            # detect video or single
            if hasattr(batch_data_samples[0], '__iter__') and hasattr(batch_data_samples[0], '__len__') \
               and hasattr(batch_data_samples[0], '__getitem__'):
                # TrackDataSample path
                meta = batch_data_samples[0][0].metainfo
            else:
                meta = batch_data_samples[0].metainfo
            h, w = meta['img_shape'][:2]
            has_text = False
            try:
                if hasattr(batch_data_samples[0], '__iter__'):
                    for det in batch_data_samples[0]:
                        if hasattr(det, 'text') and det.text:
                            has_text = True
                            break
                else:
                    has_text = hasattr(batch_data_samples[0], 'text') and bool(batch_data_samples[0].text)
            except Exception:
                pass
            # heuristic routing
            if max(h, w) >= 1280 or has_text:
                num_stages = 3
                num_queries = 100
            else:
                num_stages = 2
                num_queries = 60
            # apply to head
            if hasattr(self, 'panoptic_head'):
                self.panoptic_head.num_stages = int(num_stages)
                self.panoptic_head.num_queries = int(num_queries)
        except Exception:
            # keep defaults on any failure
            pass

    def loss(self, batch_inputs, batch_data_samples):
        # apply routing before forward
        self._apply_routing(batch_inputs, batch_data_samples)
        return super().loss(batch_inputs, batch_data_samples)

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        # apply routing before predict
        self._apply_routing(batch_inputs, batch_data_samples)
        return super().predict(batch_inputs, batch_data_samples, rescale=rescale)
