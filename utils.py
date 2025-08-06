import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from timeit import default_timer as timer
from torch.utils.data.sampler import *
import torch.nn.functional as F
import os
import shutil
import sys
import numpy as np
from cmfl import CMFL


def save(list_or_dict, name):
    f = open(name, 'w')
    f.write(str(list_or_dict))
    f.close()


def load(name):
    f = open(name, 'r')
    a = f.read()
    tmp = eval(a)
    f.close()
    return tmp


def acc(preds, targs, th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds == targs).float().mean()


def dot_numpy(vector1, vector2, emb_size=512):
    vector1 = vector1.reshape([-1, emb_size])
    vector2 = vector2.reshape([-1, emb_size])
    vector2 = vector2.transpose(1, 0)
    cosV12 = np.dot(vector1, vector2)
    return cosV12


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def softmax_cross_entropy_criterion(logit, truth, is_average=True):
    loss = F.cross_entropy(logit, truth, reduce=is_average)
    return loss

def compute_loss(a_logit, c_logit, d_logit, i_logit, truth, device):
    """
    计算给定网络、数据和标签以及计算将在其上执行的设备的损失。
    """
    criterion_cmfl = CMFL(alpha=1, gamma=3, binary=False, multiplier=2)
    criterion_bce = nn.BCELoss()

    # 设置两种损失之间的平衡系数
    beta = 0.5

    # logit, c_logit, d_logit, i_logit, truth = logit.to(device), c_logit.to(device), d_logit.to(device), i_logit.to(
    #     device), truth.to(device)

    # 调整标签的形状以匹配模型输出
    # truth = truth.view(logit.shape[0], -1).float()

    # 将图像数据转移到指定的设备并包装成Variable（注：在较新版本的PyTorch中，直接使用Tensor即可，不必使用Variable）
    # imagesv = Variable(img['image'].to(device))

    # 将二分类目标标签转移到指定的设备并包装成Variable，同样，在较新版本的PyTorch中，直接使用Tensor即可
    # labelsv_binary = Variable(labels['binary_target'].to(device))

    # 使用网络进行前向传播，得到四个输出：gap、op、op_rgb、op_d
    # 假设op为最终分类输出，op_rgb和op_d为两种不同模态的输出
    # gap, op, op_rgb, op_d = network(imagesv)

    # 使用CMFL损失函数计算基于op_rgb和op_d的损失，这里利用了模型对两种不同模态的预测输出
    # loss_cmfl = criterion_cmfl(op_rgb, op_d, labelsv_binary.unsqueeze(1).float())

    loss_cmfl = criterion_cmfl(c_logit.float(), d_logit.float(), truth.float())

    # 使用二元交叉熵损失函数计算基于op的损失，op是模型的最终分类预测输出
    # loss_bce = criterion_bce(op, labelsv_binary.unsqueeze(1).float())
    loss_bce = criterion_bce(a_logit.float(), truth.float())


    # 计算总损失，通过参数beta来平衡CMFL损失和二元交叉熵损失
    loss = beta * loss_cmfl + (1 - beta) * loss_bce

    return loss


def bce_criterion(logit, truth, is_average=True):
    loss = F.binary_cross_entropy_with_logits(logit, truth, reduce=is_average)
    return loss


def remove_comments(lines, token='#'):
    """ Generator. Strips comments and whitespace from input lines.
    """
    l = []
    for line in lines:
        s = line.split(token, 1)[0].strip()
        if s != '':
            l.append(s)
    return l


def remove(file):
    if os.path.exists(file): os.remove(file)


def empty(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir, ignore_errors=True)
    else:
        os.makedirs(dir)


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message: is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)
    else:
        raise NotImplementedError


def np_float32_to_uint8(x, scale=255.0):
    return (x * scale).astype(np.uint8)


def np_uint8_to_float32(x, scale=255.0):
    return (x / scale).astype(np.float32)


import matplotlib.pyplot as plt
import re
import os


def extract_and_plot_last_cycle_data(file_path):
    # 检查文件是否存在
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return

    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 初始化数据列表
    last_cycle_data = {
        'valid_loss_0': [],
        'valid_loss_4': [],
        'batch_loss_0': [],
        'batch_loss_1': [],
    }

    # 寻找最后一个训练周期的数据
    for line in lines:
        # 检查是否是训练记录行
        if line.startswith(model_name) and "Cycle" in line:
            # 使用正则表达式提取数据
            match = re.search(
                r"Cycle (\d+): \d+\.\d+ \d+ \d+\.\d+ \| (\d+\.\d+) (\d+\.\d+) (\d+\.\d+) \* \| (\d+\.\d+) (\d+\.\d+)",
                line)
            if match:
                cycle_index = int(match.group(1))
                valid_loss_0 = float(match.group(2))
                valid_loss_4 = float(match.group(3))
                batch_loss_0 = float(match.group(4))
                batch_loss_1 = float(match.group(5))

                # 存储数据
                last_cycle_data['valid_loss_0'].append((cycle_index, valid_loss_0))
                last_cycle_data['valid_loss_4'].append((cycle_index, valid_loss_4))
                last_cycle_data['batch_loss_0'].append((cycle_index, batch_loss_0))
                last_cycle_data['batch_loss_1'].append((cycle_index, batch_loss_1))

    # 绘制图表
    plt.figure(figsize=(12, 6))

    # 绘制valid_loss_0
    plt.subplot(4, 4, 1)
    plt.scatter([x[0] for x in last_cycle_data['valid_loss_0']], [x[1] for x in last_cycle_data['valid_loss_0']],
                label='Valid Loss 0')
    plt.title('Valid Loss 0 over Cycle')
    plt.xlabel('Cycle Index')
    plt.ylabel('Valid Loss 0')
    plt.legend()

    # 绘制valid_loss_4
    plt.subplot(2, 2, 2)
    plt.scatter([x[0] for x in last_cycle_data['valid_loss_4']], [x[1] for x in last_cycle_data['valid_loss_4']],
                label='Valid Loss 4')
    plt.title('Valid Loss 4 over Cycle')
    plt.xlabel('Cycle Index')
    plt.ylabel('Valid Loss 4')
    plt.legend()

    # 绘制batch_loss_0
    plt.subplot(2, 2, 3)
    plt.scatter([x[0] for x in last_cycle_data['batch_loss_0']], [x[1] for x in last_cycle_data['batch_loss_0']],
                label='Batch Loss 0')
    plt.title('Batch Loss 0 over Cycle')
    plt.xlabel('Cycle Index')
    plt.ylabel('Batch Loss 0')
    plt.legend()

    # 绘制batch_loss_1
    plt.subplot(2, 2, 4)
    plt.scatter([x[0] for x in last_cycle_data['batch_loss_1']], [x[1] for x in last_cycle_data['batch_loss_1']],
                label='Batch Loss 1')
    plt.title('Batch Loss 1 over Cycle')
    plt.xlabel('Cycle Index')
    plt.ylabel('Batch Loss 1')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    file_content = """
    ** net setting **
    ...
    FaceBagNet_color_48 Cycle 1: 0.0352 1823.0    2.0 | 0.330051  0.142259  0.882    | 0.320405  0.859375 | 0 hr 33 min 
    ...
    """
    model_name = 'FaceBagNet_color_48'  # 假设模型名称是固定的

    extract_and_plot_last_cycle_data('./Models/FaceBagNet_color_48/FaceBagNet_color_48.txt')
