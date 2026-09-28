# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
from jittor_compat import torch
import ocnn
import numpy as np
from tqdm import tqdm
from thsolver import Solver
import math
from tools.CylinderDrawer import CylinderVisualizer

from datasets import (get_seg_shapenet_dataset, get_scannet_dataset,
                      get_kitti_dataset)
import pdb
import time
from sklearn.metrics import f1_score
class SegSolver(Solver):

    def get_model(self, flags):
        if flags.name.lower() == 'segnet':
            model = ocnn.models.SegNet(
                flags.channel, flags.nout, flags.stages, flags.interp, flags.nempty)
        elif flags.name.lower() == 'unet':
            model = ocnn.models.UNet(
                flags.channel, flags.nout, flags.interp, flags.nempty)
        else:
            raise ValueError
        return model

    def get_dataset(self, flags):
        if flags.name.lower() == 'shapenet':
            return get_seg_shapenet_dataset(flags)
        elif flags.name.lower() == 'scannet':
            return get_scannet_dataset(flags)
        elif flags.name.lower() == 'kitti':
            return get_kitti_dataset(flags)
        else:
            raise ValueError

    def get_input_feature(self, octree):
        flags = self.FLAGS.MODEL
        octree_feature = ocnn.modules.InputFeature(flags.feature, flags.nempty)
        data = octree_feature(octree)
        return data

    def process_batch(self, batch, flags):
        def points2octree(points):
            octree = ocnn.octree.Octree(flags.depth, flags.full_depth)
            octree.build_octree(points)
            return octree

        if 'octree' in batch:
            batch['octree'] = batch['octree'].cuda(non_blocking=True)
            batch['points'] = batch['points'].cuda(non_blocking=True)
            # tool_params = batch['tool_params'].cuda(non_blocking=True)
            # batch['tool_params'] = tool_params
        else:
            points = [pts.cuda(non_blocking=True) for pts in batch['points']]
            octrees = [points2octree(pts) for pts in points]
            octree = ocnn.octree.merge_octrees(octrees)
            octree.construct_all_neigh()
            batch['points'] = ocnn.octree.merge_points(points)
            batch['octree'] = octree
            # tool_params = batch['tool_params'].cuda(non_blocking=True)
            # batch['tool_params'] = tool_params
        return batch


    def model_forward(self, batch):
        octree, points = batch['octree'], batch['points']
        #print(octree, points)
        data = self.get_input_feature(octree)
        query_pts = torch.cat([points.points, points.batch_id], dim=1)
        #print(query_pts.shape)
        # 从 batch 中提取刀具参数
        tool_params = batch['tool_params']  # 获取刀具参数
        # print(f"Original tool_params: {tool_params}, type: {type(tool_params)}")
        tool_params = [[float(item) for item in row] for row in tool_params]
        tool_params = torch.tensor(tool_params, dtype=torch.float32).cuda() #FC: 需要标注GPU序号
        # print(f"Processed tool_params: {tool_params}, type: {type(tool_params)}, shape: {tool_params.shape}")

        # 将刀具参数传递给模型
        logits = self.model.forward(data, octree, octree.depth, query_pts, tool_params)  # 传递刀具参数
        # print(points.labels.shape)
        labels = points.labels.squeeze(1)
        label_mask = labels > self.FLAGS.LOSS.mask  # filter labels
        labels_2 = points.labels_2.squeeze(1)
        # print(labels[label_mask].shape)
        #return logit_1, logit_2, labels, labels_2
        return logits[label_mask], labels[label_mask], labels_2#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    def visualization(self, points, logit, labels,  red_folder,gt_folder):
        # 打开文件进行写入
        with open(red_folder, 'w') as obj_file:
            # 遍历logit张量的每一行
            for i in range(logit.size(0)):  # 遍历每个batch的logit
                # 如果logit第i行的第一个值大于第二个值，则处理对应的点
                if logit[i, 0] > logit[i, 1]:
                    # 获取第i个batch的points
                    batch_points = points[i]

                    # 遍历该batch中的每个点
                    obj_file.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")

        with open(gt_folder, 'w') as obj_file:
            # 遍历labels张量的每一行
            for i in range(labels.size(0)):  # 遍历每个batch的labels
                # 如果labels第i行的值为0，则处理对应的点
                if labels[i] == 0:
                    batch_points = points[i]  # 获取第i个batch的points
                    # 遍历该batch中的每个点并写入到.obj文件
                    obj_file.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")
                
    def visualization1(self, points, logit, labels,  red_folder,gt_folder):
        # 打开文件进行写入
        with open(red_folder, 'w') as obj_file:
            # 遍历logit张量的每一行
            for i in range(logit.size(0)):  # 遍历每个batch的logit
                # 如果logit第i行的第一个值大于第二个值，则处理对应的点
                if logit[i, 0] < logit[i, 1]:
                    # 获取第i个batch的points
                    batch_points = points[i]

                    # 遍历该batch中的每个点
                    obj_file.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")

        with open(gt_folder, 'w') as obj_file:
            # 遍历labels张量的每一行
            for i in range(labels.size(0)):  # 遍历每个batch的labels
                # 如果labels第i行的值为0，则处理对应的点
                if labels[i] == 1:
                    batch_points = points[i]  # 获取第i个batch的points
                    # 遍历该batch中的每个点并写入到.obj文件
                    obj_file.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")

    def visualization_new(self, points, pred_points, columns, obj_folder, pred_folder, hash_ord):
        # 获取原始tensor的形状和元素总数
        original_shape = pred_points.shape
        total_elements = pred_points.numel()

        # 计算合适的行数
        if total_elements % columns != 0:
            raise ValueError(f"无法将包含{total_elements}个元素的tensor重塑为{columns}列，请检查输入tensor的维度")

        rows = total_elements // columns

        # 重塑tensor python run_seg_deepmill.py --depth 5 --model unet --alias unet_d5
        reshaped_tensor = pred_points.reshape(rows, columns)

        # 计算每行最大值的索引
        max_indices = torch.argmax(reshaped_tensor, dim=1)
        # print(max_indices.device, hash_ord.device)
        # hash_ord.to(max_indices.device)

        # print(f"原始tensor形状: {original_shape}")
        # print(f"重塑后形状: ({rows}, {columns})")
        # print(f"每行最大值索引: {max_indices}")

        # 打开文件进行写入
        # with open(obj_folder, 'w') as obj_file:
        #     # 遍历logit张量的每一行
        #     for i in range(rows):  # 遍历每个batch的logit
        #         # 获取第i个batch的points
        #         batch_points = points[i]
        #         # 遍历该batch中的每个点
        #         obj_file.write(f"vn {batch_points.normals[0]} {batch_points.normals[1]} {batch_points.normals[2]}\n")
        #         obj_file.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")


        with open(pred_folder, 'w') as obj_file:
            # 遍历logit张量的每一行
            for i in range(rows):  # 遍历每个batch的logit
                # 获取第i个batch的points
                batch_points = points[i]

                sample_point = 0.03 * hash_ord[max_indices[i]]
                obj_file.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")
                # norm = torch.norm(hash_ord[max_indices[i]], p=2)
                # hash_ord[max_indices[i]] = hash_ord[max_indices[i]]/norm
                # print(torch.norm(hash_ord[max_indices[i]], p=2))
                obj_file.write(f"v {batch_points.points[0]+sample_point[0]} {batch_points.points[1]+sample_point[1]} {batch_points.points[2]+sample_point[2]}\n")

        with open(pred_folder, 'a') as obj_file:
            # 遍历logit张量的每一行
            for i in range(1, rows+1):  # 遍历每个batch的logit
                obj_file.write(f"l {2*i-1} {2*i}\n")

    def direction_visual(self, pred_points, columns, direction_folder):
        # 获取原始tensor的形状和元素总数
        original_shape = pred_points.shape
        total_elements = pred_points.numel()

        # 计算合适的行数
        if total_elements % columns != 0:
            raise ValueError(f"无法将包含{total_elements}个元素的tensor重塑为{columns}列，请检查输入tensor的维度")

        rows = total_elements // columns

        # 重塑tensor python run_seg_deepmill.py --depth 5 --model unet --alias unet_d5
        reshaped_tensor = pred_points.reshape(rows, columns)
        reshaped_tensor = (reshaped_tensor>0.5).int()
        reshaped_tensor = reshaped_tensor.transpose(0, 1)

        # 计算每行最大值的索引
        # max_indices = torch.argmax(reshaped_tensor, dim=1)

        # 打开文件进行写入
        with open(direction_folder, 'w') as f:
            for row in reshaped_tensor:
                # 将每行数据转换为字符串，用空格分隔
                line = ' '.join(map(str, row.tolist()))
                f.write(line + '\n')

    def col_visual(self, points, pred_points, columns, col_folder):
        # 获取原始tensor的形状和元素总数
        original_shape = pred_points.shape
        total_elements = pred_points.numel()

        # 计算合适的行数
        if total_elements % columns != 0:
            raise ValueError(f"无法将包含{total_elements}个元素的tensor重塑为{columns}列，请检查输入tensor的维度")

        rows = total_elements // columns

        # 重塑tensor python run_seg_deepmill.py --depth 5 --model unet --alias unet_d5
        reshaped_tensor = pred_points.reshape(rows, columns)
        reshaped_tensor = (reshaped_tensor>0.5).int()

        # 计算每行最大值的索引
        max_indices = torch.argmax(reshaped_tensor, dim=1)

        # 打开文件进行写入
        with open(col_folder, 'w') as f:
            for i in range(rows):
                row = reshaped_tensor[i]
                if all(value == 0 for value in row.tolist()):
            # 获取第i个batch的points
                    batch_points = points[i]
                    # 遍历该batch中的每个点
                    f.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")



    def build_hash_tensor(self):
        """
        将文本向量数据转换为PyTorch张量

        Args:
            vecs_text: 包含三维向量数据的文本

        Returns:
            torch.Tensor: 形状为[102, 3]的张量
        """
        vecs_text = """
            0 0 1
            0.0998749 -0.995 0
            -0.127236 -0.985 0.116559
            0.159563 -0.965 0.208122
            -0.092068 -0.935 0.342489
            0.378973 -0.915 0.1384
            -0.393227 -0.905 0.162319
            0.139344 -0.885 0.44425
            -0.298278 -0.855 0.42427
            0.420748 -0.835 0.354607
            -0.56465 -0.825 0.0233504
            -0.0274038 -0.805 0.592642
            0.613963 -0.785 0.0826074
            -0.51875 -0.775 0.360934
            0.326014 -0.755 0.568937
            -0.265898 -0.725 0.635353
            0.627783 -0.705 0.329944
            -0.695261 -0.695 0.183269
            0.12497 -0.675 0.727157
            -0.51894 -0.645 0.56096
            0.524491 -0.625 0.578173
            -0.143852 -0.595 0.790747
            0.789058 -0.575 0.216247
            -0.734071 -0.565 0.376715
            0.316128 -0.545 0.776555
            -0.433413 -0.515 0.739546
            0.708363 -0.495 0.503186
            -0.8678 -0.485 0.108154
            0.0322655 -0.465 0.884723
            0.894362 -0.445 0.0457273
            -0.694141 -0.435 0.573536
            0.51598 -0.415 0.749359
            -0.28479 -0.385 0.877878
            0.857067 -0.365 0.363609
            -0.881226 -0.355 0.312115
            0.235717 -0.335 0.912257
            -0.586792 -0.305 0.7501
            0.70044 -0.285 0.654339
            -0.0939264 -0.255 0.962368
            0.95544 -0.235 0.178631
            -0.827182 -0.225 0.514922
            0.442625 -0.205 0.872959
            -0.425983 -0.175 0.887645
            0.849883 -0.155 0.503661
            -0.96838 -0.145 0.203015
            0.117901 -0.125 0.985127
            -0.712438 -0.095 0.695275
            0.631407 -0.075 0.771816
            -0.228754 -0.045 0.972444
            0.949387 -0.025 0.313114
            -0.911736 -0.015 0.410503
            0.329255 0.005 0.944228
            -0.548679 0.035 0.8353
            0.783427 0.055 0.619046
            -0.995234 0.065 0.0726883
            -0.014577 0.085 0.996274
            0.989286 0.105 0.101426
            -0.797094 0.115 0.592804
            0.519527 0.135 0.843722
            -0.351854 0.165 0.921398
            0.883911 0.185 0.429507
            -0.940041 0.195 0.279816
            0.195722 0.215 0.956801
            -0.635585 0.245 0.732125
            0.67009 0.265 0.693365
            -0.141252 0.295 0.944999
            0.922803 0.315 0.221831
            -0.827336 0.325 0.458137
            0.381164 0.345 0.857723
            -0.443239 0.375 0.814196
            0.765376 0.395 0.508108
            -0.903275 0.405 0.141665
            0.0615329 0.425 0.903099
            0.89536 0.445 0.0174683
            -0.668229 0.455 0.588596
            0.521694 0.475 0.708668
            -0.240285 0.505 0.828998
            0.793611 0.525 0.307499
            -0.787074 0.535 0.307065
            0.23343 0.555 0.798427
            -0.479302 0.585 0.654251
            0.598708 0.605 0.524903
            -0.788339 0.615 0.0172067
            -0.0507286 0.635 0.770845
            0.746773 0.655 0.115349
            -0.621256 0.665 0.414507
            0.349819 0.685 0.639063
            -0.282433 0.715 0.639536
            0.593951 0.735 0.327105
            -0.648225 0.745 0.157415
            0.0966825 0.765 0.636732
            -0.420543 0.795 0.437171
            0.380886 0.815 0.436693
            -0.105958 0.845 0.524164
            0.481249 0.865 0.142038
            -0.434947 0.875 0.212594
            0.160096 0.895 0.416346
            -0.198477 0.925 0.324009
            0.262898 0.945 0.194575
            -0.294988 0.955 0.0309341
            0.00376534 0.975 0.222173
            0.0996262 0.995 0.00704444
            """
        vectors = []
        lines = vecs_text.strip().split('\n')

        for line in lines:
            values = line.strip().split()
            if len(values) == 3:
                try:
                    x, y, z = map(float, values)
                    vectors.append([x, y, z])
                except ValueError:
                    print(f"警告: 无法解析行: {line}")

        # 转换为PyTorch张量
        hash_ord = torch.tensor(vectors, dtype=torch.float32)

        # 验证形状
        if hash_ord.shape != (102, 3):
            raise ValueError(f"张量形状应为(102, 3)，实际为{hash_ord.shape}")

        return hash_ord

    def train_step(self, batch):
        batch = self.process_batch(batch, self.FLAGS.DATA.train)
        #print(batch['points'].labels_2.shape)
        logits, label, label_2 = self.model_forward(batch)
        logits = torch.sigmoid(logits)
        #print(logits.shape, label.shape)
        loss_1 = self.loss_function(logits, label)
        loss_2 = 0.0
        loss = loss_1
        # loss = (loss_1 + loss_2)/2
        accu_1 = self.accuracy(logits, label)
        # accu_2 = self.accuracy(logit_2, label_2)
        accu = accu_1

        # pred_1 = logit_1.argmax(dim=-1)  # 假设 logit_1 是 logits 形式，需要用 argmax 选取预测类别
        # pred_2 = logit_2.argmax(dim=-1)
        pred = (logits > 0.5)
        # 这里使用 f1_score 函数，假设 label 和 label_2 都是 0 和 1 的整数标签
        f1_score_1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), average='macro')#average='binary')
        # f1_score_2 = f1_score(label_2.cpu().numpy(), pred_2.cpu().numpy(), average='macro')#average='binary')
        # f1_score_avg = (f1_score_1 + f1_score_2) / 2
        f1_score_avg = f1_score_1

        # return {'train/loss': loss, 'train/accu': accu, 'train/accu_red': accu_1, 'train/accu_green': accu_2,
        #         'train/f1_red': torch.tensor(f1_score_1, dtype=torch.float32).cuda(), 'train/f1_green': torch.tensor(f1_score_2, dtype=torch.float32).cuda(), 'train/f1_avg': torch.tensor(f1_score_avg, dtype=torch.float32).cuda()}
        return {'train/loss': loss, 'train/accu': accu,'train/accu_red': accu_1,
        'train/f1_red': torch.tensor(f1_score_1, dtype=torch.float32).cuda(),'train/f1_avg': torch.tensor(f1_score_avg, dtype=torch.float32).cuda()}



    def test_step(self, batch):
        batch = self.process_batch(batch, self.FLAGS.DATA.test)
        # print(batch['points'].points.shape)
        with torch.no_grad():
            start_time = time.time()
            logits, label, label_2 = self.model_forward(batch)
            logits = torch.sigmoid(logits)
            end_time = time.time()
            inference_time = (end_time - start_time)
            print(f"推理时间: {inference_time:.6f}秒")
            with open("./visual/time.txt", 'a') as file:
                file.write(f"{batch['filename'][0]} inference_time: {inference_time}\n")
        # print(label.shape, logits.shape)
        # self.visualization(batch['points'], logit, label, ".\\data\\vis\\"+batch['filename'][0][:-4]+".obj") #FC:目前可视化只支持test的batch size=1
        loss_1 = self.loss_function(logits, label)
        loss_2 = loss_1
        loss = (loss_1 + loss_2) / 2
        accu_1 = self.accuracy(logits, label)
        accu_2 = accu_1
        accu = (accu_1 + accu_2) / 2
        num_class = self.FLAGS.LOSS.num_class
        #IoU, insc, union = self.IoU_per_shape(logits, label, num_class)
        folders = [
            './visual/obj',
            './visual/pred',
            './visual/direction',
            './visual/col'
        ]
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
              
        # red_folder = os.path.join(r"./visual/red_points",
        #                           batch['filename'][0].split("/")[-1].split(".")[0].split("_collision_detection")[
        #                               0] + ".obj")
        # gt_red_folder = os.path.join(r"./visual/GT_red",
        #                              batch['filename'][0].split("/")[-1].split(".")[0].split("_collision_detection")[
        #                                  0] + ".obj")
        # green_folder = os.path.join(r'./visual/green_points',
        #                             batch['filename'][0].split("/")[-1].split(".")[0].split("_collision_detection")[
        #                                 0] + ".obj")
        # gt_green_folder = os.path.join(r'./visual/GT_green',
        #                                batch['filename'][0].split("/")[-1].split(".")[0].split("_collision_detection")[
        #                                    0] + ".obj")
        pred_folder = os.path.join(r'./visual/pred',
                                       batch['filename'][0].split("/")[-1].split(".")[0].split("_collision_detection")[
                                           0] + ".obj")
        obj_folder = os.path.join(r'./visual/obj',
                                  batch['filename'][0].split("/")[-1].split(".")[0].split("_collision_detection")[
                                      0] + ".obj")
        direction_folder = os.path.join(r'./visual/direction',
                                  batch['filename'][0].split("/")[-1].split(".")[0].split("_collision_detection")[
                                      0] + ".txt")
        col_folder = os.path.join(r'./visual/col',
                                        batch['filename'][0].split("/")[-1].split(".")[0].split("_collision_detection")[
                                            0] + ".obj")
        hash_ord = self.build_hash_tensor()
        # print(hash_ord[101])
        #self.visualization_new(batch['points'], logits, 102, obj_folder, pred_folder, hash_ord)

        self.direction_visual(logits, 102, direction_folder)
        # CyVisual = CylinderVisualizer(radius=0.003, segments=8)
        # CyVisual.visualization_new(batch['points'], logits, 102, obj_folder, pred_folder, hash_ord)
        # self.col_visual(batch['points'], logits, 102, col_folder)

        # self.visualization(batch['points'], logits, label, red_folder, gt_red_folder)
        # self.visualization1(batch['points'], logit_2, label_2, green_folder, gt_green_folder)
        # pred_1 = logit_1.argmax(dim=-1)
        # pred_2 = logit_2.argmax(dim=-1)
        pred = (logits>0.5)
        # torch.set_printoptions(profile="full")
        # print('\n pred:')
        # print(pred[-102:])
        # print('label:')
        # print(label[-102:])
        # print('---------------------------------------------------------')
        # 这里使用 f1_score 函数，假设 label 和 label_2 都是 0 和 1 的整数标签
        f1_score_1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), average='macro')#average='binary')
        f1_score_2 = f1_score_1
        f1_score_avg = (f1_score_1 + f1_score_2) / 2

        # names = ['test/loss', 'test/accu', 'test/accu_red','test/accu_green','test/mIoU', 'test/f1_red','test/f1_green','test/f1_avg'] + \
        #         ['test/intsc_%d' % i for i in range(num_class)] + \
        #         ['test/union_%d' % i for i in range(num_class)]
        # tensors = [loss, accu, accu_1, accu_2, IoU, torch.tensor(f1_score_1, dtype=torch.float32).cuda(),
        #            torch.tensor(f1_score_2, dtype=torch.float32).cuda(),
        #            torch.tensor(f1_score_avg, dtype=torch.float32).cuda()] + insc + union\
        names = ['test/loss', 'test/accu', 'test/accu_red', 'test/accu_green', 'test/f1_red',
                 'test/f1_green', 'test/f1_avg']
        tensors = [loss, accu, accu_1, accu_2, torch.tensor(f1_score_1, dtype=torch.float32).cuda(),
                   torch.tensor(f1_score_2, dtype=torch.float32).cuda(),
                   torch.tensor(f1_score_avg, dtype=torch.float32).cuda()]
        return dict(zip(names, tensors))


    def eval_step(self, batch):
        batch = self.process_batch(batch, self.FLAGS.DATA.test)
        with torch.no_grad():
            logit, _ = self.model_forward(batch)
        prob = torch.nn.functional.softmax(logit, dim=1)

        # split predictions
        inbox_masks = batch['inbox_mask']
        npts = batch['points'].batch_npt.tolist()
        probs = torch.split(prob, npts)

        # merge predictions
        batch_size = len(inbox_masks)
        for i in range(batch_size):
            # The point cloud may be clipped when doing data augmentation. The
            # `inbox_mask` indicates which points are clipped. The `prob_all_pts`
            # contains the prediction for all points.
            prob = probs[i].cpu()
            inbox_mask = inbox_masks[i].to(prob.device)
            prob_all_pts = prob.new_zeros([inbox_mask.shape[0], prob.shape[1]])
            prob_all_pts[inbox_mask] = prob

            # Aggregate predictions across different epochs
            filename = batch['filename'][i]
            self.eval_rst[filename] = self.eval_rst.get(filename, 0) + prob_all_pts

            # Save the prediction results in the last epoch
            if self.FLAGS.SOLVER.eval_epoch - 1 == batch['epoch']:
                full_filename = os.path.join(self.logdir, filename[:-4] + '.eval.npz')
                curr_folder = os.path.dirname(full_filename)
                if not os.path.exists(curr_folder): os.makedirs(curr_folder)
                np.savez(full_filename, prob=self.eval_rst[filename].cpu().numpy())

    def result_callback(self, avg_tracker, epoch):
        r''' Calculate the part mIoU for PartNet and ScanNet.
        '''

        iou_part = 0.0
        avg = avg_tracker.average()

        # Labels smaller than `mask` is ignored. The points with the label 0 in
        # PartNet are background points, i.e., unlabeled points
        mask = self.FLAGS.LOSS.mask + 1
        num_class = self.FLAGS.LOSS.num_class
        for i in range(mask, num_class):
            instc_i = avg['test/intsc_%d' % i]
            union_i = avg['test/union_%d' % i]
            iou_part += instc_i / (union_i + 1.0e-10)
        iou_part = iou_part / (num_class - mask)

        avg_tracker.update({'test/mIoU_part': torch.Tensor([iou_part])})
        tqdm.write('=> Epoch: %d, test/mIoU_part: %f' % (epoch, iou_part))

    def loss_function_old(self, logit, label):
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logit, label.long())
        return loss

    def loss_function(self, logit, label):
        """
        计算二分类交叉熵损失
        :param logit: sigmoid归一化后的概率值
        :param label: 真实标签，值为0或1
        :return: 标量损失值
        """
        # 添加微小值防止log(0)计算
        epsilon = 1e-7
        logit = torch.clamp(logit, epsilon, 1. - epsilon)

        # 计算交叉熵损失
        loss = - (label * torch.log(logit) + (1 - label) * torch.log(1 - logit))

        # 对batch求平均
        return torch.mean(loss)


    def accuracy_old(self, logit, label):
        pred = logit.argmax(dim=1)
        accu = pred.eq(label).float().mean()
        return accu

    def accuracy(self, logit, label):
        """
        计算二分类准确率
        :param logit: sigmoid归一化后的概率值
        :param label: 真实标签，值为0或1
        :return: 准确率标量值(0~1)
        """
        # 将概率转换为预测类别（>0.5为1，否则为0）
        pred = (logit > 0.5).float()
        # 统计预测正确的样本数
        correct = (pred == label).float().mean()
        # 计算准确率
        return correct


    def IoU_per_shape(self, logit, label, class_num):
        pred = logit.argmax(dim=1)

        IoU, valid_part_num, esp = 0.0, 0.0, 1.0e-10
        intsc, union = [None] * class_num, [None] * class_num
        for k in range(class_num):
            pk, lk = pred.eq(k), label.eq(k)
            intsc[k] = torch.sum(torch.logical_and(pk, lk).float())
            union[k] = torch.sum(torch.logical_or(pk, lk).float())

            valid = torch.sum(lk.any()) > 0
            valid_part_num += valid.item()
            IoU += valid * intsc[k] / (union[k] + esp)

        # Calculate the shape IoU for ShapeNet
        IoU /= valid_part_num + esp
        return IoU, intsc, union


if __name__ == "__main__":

    SegSolver.main()
