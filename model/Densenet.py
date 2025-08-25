import re

import torch
from torch import nn
from torch.utils import model_zoo
from torchvision.models.video.resnet import model_urls
from collections import OrderedDict
import torch.nn.functional as F

# 这里我们采用Pytorch框架来实现DenseNet，目前它已经支持Windows系统。对于DenseNet，Pytorch在torchvision.models模块里给出了官方实现，这个DenseNet版本是用于ImageNet数据集的DenseNet-BC模型，下面简单介绍实现过程。
#
# 首先实现DenseBlock中的内部结构，这里是BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv结构，最后也加入dropout层以用于训练过程。
# ————————————————
#
#                             版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。
#
# 原文链接：https://blog.csdn.net/qq_44766883/article/details/112011420
class _DenseLayer(nn.Sequential):
    """Basic unit of DenseBlock (using bottleneck layer) """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size*growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
# 实现DenseBlock模块，内部是密集连接方式（输入特征数线性增长）：
class _DenseBlock(nn.Sequential):
    """DenseBlock"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size,
                                drop_rate)
            self.add_module("denselayer%d" % (i+1,), layer)

# 此外，我们实现Transition层，它主要是一个卷积层和一个池化层：
class _Transition(nn.Sequential):
    """Transition layer between two adjacent DenseBlock"""
    def __init__(self, num_input_feature, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_feature))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_feature, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(2, stride=2))


# DenseNet网络的构造函数
class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, compression_rate=0.5, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()  # 调用父类(nn.Module)的构造函数

        # 初始化网络的第一个卷积层
        self.features = nn.Sequential(OrderedDict([
            # 使用7x7的卷积核，步长为2，填充为3，不使用偏置项
            ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),  # 批量归一化
            ("relu0", nn.ReLU(inplace=True)),  # ReLU激活函数
            ("pool0", nn.MaxPool2d(3, stride=2, padding=1))  # 最大池化层
        ]))

        # 构建DenseBlock
        num_features = num_init_features  # 初始化特征数量
        for i, num_layers in enumerate(block_config):  # 遍历每个DenseBlock的层数配置
            # 创建DenseBlock
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module("denseblock%d" % (i + 1), block)
            # 更新特征数量
            num_features += num_layers * growth_rate

            # 最后一个DenseBlock后不添加Transition Layer
            if i != len(block_config) - 1:
                # 创建Transition Layer
                transition = _Transition(num_features, int(num_features * compression_rate))
                self.features.add_module("transition%d" % (i + 1), transition)
                # 更新特征数量为压缩后的数量
                num_features = int(num_features * compression_rate)

        # 添加最后的批量归一化和ReLU激活函数
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))

        # 分类层
        self.classifier = nn.Linear(num_features, num_classes)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)  # 卷积层参数初始化
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)  # 批量归一化层偏置初始化为0
                nn.init.constant_(m.weight, 1)  # 批量归一化层权重初始化为1
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)  # 线性层偏置初始化为0

    # 定义前向传播函数
    def forward(self, x):
        features = self.features(x)  # 通过前面定义的层传播
        # 对特征图进行平均池化，然后调整形状以适应线性层
        out = F.avg_pool2d(features, 7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)  # 通过分类层得到最终输出
        return out




import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict

class Net(nn.Module):
    def __init__(self, num_classes=2, is_first_bn=False, type="A"):
        """
        网络构造函数
        :param num_classes: 分类任务的类别数
        :param is_first_bn: 是否在网络的第一层添加批量归一化
        :param type: 选择不同的网络类型（'A', 'B', 'C', 'baseline'）
        """
        super(Net, self).__init__()

        self.is_first_bn = is_first_bn

        # 如果选择在第一层添加批量归一化，则初始化批量归一化层
        if self.is_first_bn:
            self.first_bn = nn.BatchNorm2d(3)

        # 根据type的值选择不同的encoder
        if type == 'A':
            self.encoder = models.densenet121(pretrained=True)
        elif type == 'B':
            self.encoder = models.densenet161(pretrained=True)
        elif type == 'C':
            self.encoder = models.densenet169(pretrained=True)
        elif type == 'baseline':
            self.encoder = models.densenet201(pretrained=True)

        # 注意：这里假设我们使用DenseNet的特征提取部分，而不是整个网络
        self.features = self.encoder.features

        # 根据DenseNet版本的不同，最后的特征数量也会有所不同
        # 例如，DenseNet121最后的特征数量是1024
        num_features = 1024  # 这个值应根据选择的DenseNet调整

        # 定义一个全连接层，用于将编码器的特征映射为分类结果
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入的图像数据
        """
        batch_size = x.size(0)

        if self.is_first_bn:
            x = self.first_bn(x)

        # 数据通过DenseNet的特征提取部分
        x = self.features(x)

        # 应用全局平均池化并调整形状
        fea = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, -1)

        # 通过全连接层得到最终的分类结果
        logit = self.classifier(fea)

        return logit

    def forward_features(self, x):
        """
        提取特征的函数
        :param x: 输入的图像数据
        """
        # 数据通过各个卷积层
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)

        # 假设我们需要从第一个denseblock后提取特征
        x = self.features.denseblock1(x)
        # 提取特征后，可以选择不再继续前向传播
        # 返回提取的特征
        return x

    def set_mode(self, mode, is_freeze_bn=False):
        """
        设置模型的模式，包括冻结批量归一化层
        :param mode: 模式字符串，可以是'eval'，'train'等
        :param is_freeze_bn: 是否冻结批量归一化层的参数
        """
        if mode == 'eval':
            self.eval()  # 设置为评估模式
            if is_freeze_bn:
                # 冻结所有的批量归一化层
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        elif mode == 'train':
            self.train()  # 设置为训练模式
            if is_freeze_bn:
                # 冻结所有的批量归一化层
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
            else:
                # 解冻所有的批量归一化层
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.train()
                        m.weight.requires_grad = True
                        m.bias.requires_grad = True



def densenet121(pretrained=False, **kwargs):
    """DenseNet121"""
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)

    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model

if __name__ == '__main__':
    model = densenet121()
    x = torch.rand(size=(1, 3, 224, 224))
    print(model(x).shape)
