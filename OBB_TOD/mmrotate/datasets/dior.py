import BboxToolkit as bt

import copy
import mmcv
import numpy as np

from mmdet.datasets.custom import CustomDataset

from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np

from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class DIORDataset(CustomDataset):

    CLASSES = bt.get_classes('dior')

    def __init__(self,
                 xmltype,
                 imgset,
                 ann_file,
                 img_prefix,
                 *args,
                 **kwargs):
        assert xmltype in ['hbb', 'obb']
        self.xmltype = xmltype
        self.imgset = imgset
        super(DIORDataset, self).__init__(*args,
                                          ann_file=ann_file,
                                          img_prefix=img_prefix,
                                          **kwargs)
    @classmethod
    def get_classes(cls, classes=None):
        if classes is None:
            cls.custom_classes = False
            return cls.CLASSES

        cls.custom_classes = True
        return bt.get_classes(classes)

    def load_annotations(self, ann_file):
        contents, _ = bt.load_dior(
            img_dir=self.img_prefix,
            ann_dir=ann_file,
            classes=self.CLASSES,
            xmltype=self.xmltype)
        if self.imgset is not None:
            contents = bt.split_imgset(contents, self.imgset)
        return contents

    def pre_pipeline(self, results):
        results['cls'] = self.CLASSES
        super().pre_pipeline(results)

    def format_results(self, results, save_dir=None, **kwargs):
        assert len(results) == len(self.data_infos)
        contents = []
        for result, data_info in zip(results, self.data_infos):
            info = copy.deepcopy(data_info)
            info.pop('ann')

            ann, bboxes, labels, scores = dict(), list(), list(), list()
            for i, dets in enumerate(result):
                bboxes.append(dets[:, :-1])
                scores.append(dets[:, -1])
                labels.append(np.zeros((dets.shape[0], ), dtype=np.int) + i)
            ann['bboxes'] = np.concatenate(bboxes, axis=0)
            ann['labels'] = np.concatenate(labels, axis=0)
            ann['scores'] = np.concatenate(scores, axis=0)
            info['ann'] = ann
            contents.append(info)

        if save_dir is not None:
            bt.save_pkl(save_dir, contents, self.CLASSES)
        return contents

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        else:
            raise NotImplementedError

        return eval_results