# Code inherited from MMDetection 3.x
# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from typing import Iterator, List, Optional, Sized, Union

import numpy as np
import torch
from mmdet.datasets import ConcatDataset
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler

from ..builder import SAMPLERS as DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class MultiSourceSampler(Sampler):
    r"""Multi-Source Infinite Sampler.

    According to the sampling ratio, sample data from different
    datasets to form batches.

    Args:
        dataset (Sized): The dataset.
        samples_per_gpu (int): Size of mini-batch.
        sample_ratio (list[int | float]): The sampling ratio of different
            source datasets in a mini-batch.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, set a random seed.
            Defaults to None.

    Examples:
        >>> dataset_type = 'ConcatDataset'
        >>> sub_dataset_type = 'CocoDataset'
        >>> data_root = 'data/coco/'
        >>> sup_ann = '../coco_semi_annos/instances_train2017.1@10.json'
        >>> unsup_ann = '../coco_semi_annos/' \
        >>>             'instances_train2017.1@10-unlabeled.json'
        >>> dataset = dict(type=dataset_type,
        >>>     datasets=[
        >>>         dict(
        >>>             type=sub_dataset_type,
        >>>             data_root=data_root,
        >>>             ann_file=sup_ann,
        >>>             data_prefix=dict(img='train2017/'),
        >>>             filter_cfg=dict(filter_empty_gt=True, min_size=32),
        >>>             pipeline=sup_pipeline),
        >>>         dict(
        >>>             type=sub_dataset_type,
        >>>             data_root=data_root,
        >>>             ann_file=unsup_ann,
        >>>             data_prefix=dict(img='train2017/'),
        >>>             filter_cfg=dict(filter_empty_gt=True, min_size=32),
        >>>             pipeline=unsup_pipeline),
        >>>         ])
        >>>     train_dataloader = dict(
        >>>         samples_per_gpu=5,
        >>>         num_workers=5,
        >>>         persistent_workers=True,
        >>>         sampler=dict(type='MultiSourceSampler',
        >>>             samples_per_gpu=5, sample_ratio=[1, 4]),
        >>>         batch_sampler=None,
        >>>         dataset=dataset)
    """

    def __init__(self,
                 dataset: Sized,
                 samples_per_gpu: int,
                 sample_ratio: List[Union[int, float]],
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:

        assert hasattr(dataset, 'cumulative_sizes'),\
            f'The dataset must be ConcatDataset, but get {dataset}'
        assert isinstance(samples_per_gpu, int) and samples_per_gpu > 0, \
            'samples_per_gpu must be a positive integer value, ' \
            f'but got samples_per_gpu={samples_per_gpu}'
        assert isinstance(sample_ratio, list), \
            f'sample_ratio must be a list, but got sample_ratio={sample_ratio}'
        assert len(sample_ratio) == len(dataset.cumulative_sizes), \
            'The length of sample_ratio must be equal to ' \
            f'the number of datasets, but got sample_ratio={sample_ratio}'

        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.cumulative_sizes = [0] + dataset.cumulative_sizes
        self.samples_per_gpu = samples_per_gpu
        self.sample_ratio = sample_ratio
        self.num_per_source = [
            int(samples_per_gpu * sr / sum(sample_ratio)) for sr in sample_ratio
        ]
        self.num_per_source[0] = samples_per_gpu - sum(self.num_per_source[1:])

        assert sum(self.num_per_source) == samples_per_gpu, \
            'The sum of num_per_source must be equal to ' \
            f'samples_per_gpu, but get {self.num_per_source}'

        self.seed = seed
        self.shuffle = shuffle
        self.source2inds = {
            source: self._indices_of_rank(len(ds))
            for source, ds in enumerate(dataset.datasets)
        }

    def _infinite_indices(self, sample_size: int) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(sample_size, generator=g).tolist()
            else:
                yield from torch.arange(sample_size).tolist()

    def _indices_of_rank(self, sample_size: int) -> Iterator[int]:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(
            self._infinite_indices(sample_size), self.rank, None,
            self.world_size)

    def __iter__(self) -> Iterator[int]:
        batch_buffer = []
        while True:
            for source, num in enumerate(self.num_per_source):
                batch_buffer_per_source = []
                for idx in self.source2inds[source]:
                    idx += self.cumulative_sizes[source]
                    batch_buffer_per_source.append(idx)
                    if len(batch_buffer_per_source) == num:
                        batch_buffer += batch_buffer_per_source
                        break
            yield from batch_buffer
            batch_buffer = []

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        """Not supported in `epoch-based runner."""
        pass


@DATA_SAMPLERS.register_module()
class GroupMultiSourceSampler(MultiSourceSampler):
    r"""Group Multi-Source Infinite Sampler.

    According to the sampling ratio, sample data from different
    datasets but the same group to form batches.

    Args:
        dataset (Sized): The dataset.
        samples_per_gpu (int): Size of mini-batch.
        sample_ratio (list[int | float]): The sampling ratio of different
            source datasets in a mini-batch.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, set a random seed.
            Defaults to None.
    """

    def __init__(self,
                 dataset: ConcatDataset,
                 samples_per_gpu: int,
                 sample_ratio: List[Union[int, float]],
                 shuffle: bool = True,
                 seed: Optional[int] = 0) -> None:
        super().__init__(
            dataset=dataset,
            samples_per_gpu=samples_per_gpu,
            sample_ratio=sample_ratio,
            shuffle=shuffle,
            seed=seed)

        self._get_source_group_info()
        self.group_source2inds = [{
            source:
            self._indices_of_rank(self.group2size_per_source[source][group])
            for source in range(len(dataset.datasets))
        } for group in range(len(self.group_ratio))]

    def _get_source_group_info(self) -> None:
        self.group2size_per_source = [{0: 0, 1: 0}, {0: 0, 1: 0}]
        self.group2inds_per_source = [{0: [], 1: []}, {0: [], 1: []}]
        for source, dataset in enumerate(self.dataset.datasets):
            for idx in range(len(dataset)):
                # data_info = dataset.get_data_info(idx)
                # width, height = data_info['width'], data_info['height']
                # group = 0 if width < height else 1
                group = dataset.flag[idx]
                self.group2size_per_source[source][group] += 1
                self.group2inds_per_source[source][group].append(idx)

        self.group_sizes = np.zeros(2, dtype=np.int64)
        for group2size in self.group2size_per_source:
            for group, size in group2size.items():
                self.group_sizes[group] += size
        self.group_ratio = self.group_sizes / sum(self.group_sizes)

    def __iter__(self) -> Iterator[int]:
        batch_buffer = []
        while True:
            group = np.random.choice(
                list(range(len(self.group_ratio))), p=self.group_ratio)
            for source, num in enumerate(self.num_per_source):
                batch_buffer_per_source = []
                for idx in self.group_source2inds[group][source]:
                    idx = self.group2inds_per_source[source][group][
                        idx] + self.cumulative_sizes[source]
                    batch_buffer_per_source.append(idx)
                    if len(batch_buffer_per_source) == num:
                        batch_buffer += batch_buffer_per_source
                        break
            yield from batch_buffer
            batch_buffer = []
