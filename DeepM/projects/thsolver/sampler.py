# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

from jittor_compat import torch
from typing import Any as Dataset


class InfSampler:

  def __init__(self, dataset: Dataset, shuffle: bool = True) -> None:
    self.dataset = dataset
    self.shuffle = shuffle
    self.reset_sampler()

  def reset_sampler(self):
    num = len(self.dataset)
    indices = torch.randperm(num) if self.shuffle else torch.arange(num)
    self.indices = indices.tolist()
    self.iter_num = 0

  def __iter__(self):
    return self

  def __next__(self):
    value = self.indices[self.iter_num]
    self.iter_num = self.iter_num + 1

    if self.iter_num >= len(self.indices):
      self.reset_sampler()
    return value

  def __len__(self):
    return len(self.dataset)


class DistributedInfSampler(InfSampler):

  def __init__(self, dataset: Dataset, shuffle: bool = True) -> None:
    super().__init__(dataset, shuffle=shuffle)

  def reset_sampler(self):
    raise NotImplementedError('Distributed sampling is not available in this Jittor port.')

  def __iter__(self):
    return self

  def __next__(self):
    value = self.indices[self.iter_num]
    self.iter_num = self.iter_num + 1

    if self.iter_num >= len(self.indices):
      self.reset_sampler()
    return value
