# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import time
from jittor_compat import torch
from jittor_compat import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Optional


class AverageTracker:

  def __init__(self):
    self.value = dict()
    self.num = dict()
    self.max_len = 76
    self.tick = time.time()
    self.start_time = time.time()

  def update(self, value: Dict[str, torch.Tensor], record_time: bool = True):
    r'''Update the tracker with the given value, which is called at the end of
    each iteration.
    '''

    if not value:
      return    # empty input, return

    # roughly record the update time
    if record_time:
      curr_time = time.time()
      value['time/iter'] = torch.Tensor([curr_time - self.tick])
      self.tick = curr_time

    # update the value and num
    for key, val in value.items():
      # 转换为PyTorch张量
      # print(key, type(val))
      #tensor_val = torch.from_numpy(val).float()
      self.value[key] = self.value.get(key, 0) + val.detach()
      self.num[key] = self.num.get(key, 0) + 1

  def average(self):
    return {key: val.item() / self.num[key] for key, val in self.value.items()}

  @torch.no_grad()
  def average_all_gather(self):
    r'''Average the tensors on all GPUs using all_gather, which is called at the
    end of each epoch.
    '''

    raise NotImplementedError('Distributed all-gather is not available in this Jittor port.')

  def log(self, epoch: int, summary_writer: Optional[SummaryWriter] = None,
          log_file: Optional[str] = None, msg_tag: str = '->', notes: str = '',
          print_time: bool = True, print_memory: bool = False):
    r'''Log the average value to the console, tensorboard and log file.
    '''
    if not self.value:
      return  # empty, return

    avg = self.average()
    msg = 'Epoch: %d' % epoch
    for key, val in avg.items():
      msg += ', %s: %.3f' % (key, val)
      if summary_writer is not None:
        summary_writer.add_scalar(key, val, epoch)

    # if the log_file is provided, save the log
    if log_file is not None:
      with open(log_file, 'a') as fid:
        fid.write(msg + '\n')

    # memory
    memory = ''
    if print_memory and torch.cuda.is_available():
      size = torch.cuda.memory_reserved()
      # size = torch.cuda.memory_allocated()
      memory = ', memory: {:.3f}GB'.format(size / 2**30)

    # time
    time_str = ''
    if print_time:
      curr_time = ', time: ' + datetime.now().strftime("%Y/%m/%d %H:%M:%S")
      duration = ', duration: {:.2f}s'.format(time.time() - self.start_time)
      time_str = curr_time + duration

    # other notes
    if notes:
      notes = ', ' + notes

    # concatenate all messages
    msg += memory + time_str + notes

    # split the msg for better display
    chunks = [msg[i:i+self.max_len] for i in range(0, len(msg), self.max_len)]
    msg = (msg_tag + ' ') + ('\n' + len(msg_tag) * ' ' + ' ').join(chunks)
    tqdm.write(msg)
