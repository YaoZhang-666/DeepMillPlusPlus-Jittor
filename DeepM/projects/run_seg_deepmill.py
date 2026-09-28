# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import math
import argparse
import numpy as np
import  pdb
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, default='unet_d5')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--depth', type=int, default=5)
parser.add_argument('--model', type=str, default='unet')
parser.add_argument('--mode', type=str, default='randinit')
parser.add_argument('--ckpt', type=str, default='')
parser.add_argument('--ratios', type=float, default=[1], nargs='*')

args = parser.parse_args()

alias = args.alias
gpu = args.gpu
ratios = args.ratios

module = 'segmentation.py'
data = 'data'
logdir = 'logs/seg_deepmill'

categories = ['models']
names = ['models']
seg_num = [102]
train_num = [8942]
test_num = [2236]
max_epoches = [100]
test_every_epoch = 5

max_iters = [1500]#1500

for ratio in ratios:
    for k, cat in enumerate(categories):

        mul = 2 if ratio < 0.1 else 1
        max_epoch = int(max_epoches[k] * ratio * mul)
        milestone1, milestone2 = int(0.5 * max_epoch), int(0.25 * max_epoch)
        take = int(math.ceil(train_num[k] * ratio))

        logs = os.path.join(
            logdir, f'{alias}/{cat}_{names[k]}/ratio_{ratio:.2f}'
        )

        cmd = [
            "python", module,
            "--config", "configs/seg_deepmill.yaml",
            "SOLVER.gpu", f"{gpu},",
            "SOLVER.logdir", logs,
            "SOLVER.max_epoch", str(max_epoch),
            "SOLVER.milestones", f"{milestone1},{milestone2}",
            "SOLVER.test_every_epoch", str(test_every_epoch),
            "SOLVER.ckpt", args.ckpt,
            "DATA.train.depth", str(args.depth),
            "DATA.train.filelist", f"{data}/filelist/{cat}_train_val.txt",
            "DATA.train.take", str(take),
            "DATA.test.depth", str(args.depth),
            "DATA.test.filelist", f"{data}/filelist/{cat}_test.txt",
            "MODEL.stages", str(args.depth - 2),
            "MODEL.nout", str(seg_num[k]),
            "MODEL.name", args.model,
            "LOSS.num_class", str(seg_num[k]),
        ]

        print("\n>>> Running command:\n", " ".join(cmd), "\n")

        subprocess.run(cmd)

summary = []
summary.append('names, ' + ', '.join(names) + ', C.mIoU, I.mIoU')
summary.append('train_num, ' + ', '.join([str(x) for x in train_num]))
summary.append('test_num, ' + ', '.join([str(x) for x in test_num]))

for i in range(len(ratios)-1, -1, -1):
  ious = [None] * len(categories)
  for j in range(len(categories)):
    filename = '{}/{}/{}_{}/ratio_{:.2f}/log.csv'.format(
        logdir, alias, categories[j], names[j], ratios[i])
    with open(filename, newline='') as fid:
      lines = fid.readlines()
    last_line = lines[-1]

  #   pos = last_line.find('test/mIoU:')
  #   ious[j] = float(last_line[pos+11:pos+16])
  # CmIoU = np.array(ious).mean()
  # ImIoU = np.sum(np.array(ious)*np.array(test_num)) / np.sum(np.array(test_num))
  #
  # ious = [str(iou) for iou in ious] + \
  #        ['{:.3f}'.format(CmIoU), '{:.3f}'.format(ImIoU)]
  # summary.append('Ratio:{:.2f}, '.format(ratios[i]) + ', '.join(ious))

with open('{}/{}/summaries.csv'.format(logdir, alias), 'w') as fid:
  summ = '\n'.join(summary)
  fid.write(summ)
  print(summ)

with open("./visual/time.txt", 'a') as file:
  file.write(f"======================================\n")
