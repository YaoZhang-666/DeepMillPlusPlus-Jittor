# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import numpy as np
from typing import Optional
from plyfile import PlyData, PlyElement


def save_points_to_ply(filename: str, points: np.ndarray,
                       normals: Optional[np.ndarray] = None,
                       colors: Optional[np.ndarray] = None,
                       labels: Optional[np.ndarray] = None,
                      labels_2: Optional[np.ndarray] = None,
                       text: bool = False):
  # 基础字段定义
  point_cloud = [points]
  point_cloud_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

  # 添加法向量字段
  if normals is not None:
    point_cloud.append(normals)
    point_cloud_types += [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]

  # 添加颜色字段
  if colors is not None:
    point_cloud.append(colors)
    point_cloud_types += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

  # 动态生成标签字段
  if labels is not None:
    point_cloud.append(labels)
    point_cloud_types += [(f'label_{i}', 'f4') for i in range(labels.shape[1])]

  if labels_2 is not None:
    point_cloud.append(labels_2)
    point_cloud_types += [(f'label2_{i}', 'f4') for i in range(labels_2.shape[1])]

  # 合并数据并转换格式
  point_cloud = np.concatenate(point_cloud, axis=1)
  vertices = [tuple(p) for p in point_cloud]
  structured_array = np.array(vertices, dtype=point_cloud_types)

  # 创建目录并保存
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  PlyData([PlyElement.describe(structured_array, 'vertex')], text=text).write(filename)




