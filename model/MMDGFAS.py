"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
# from torch._dynamo.backends import tvm

from SKnet.SKNet import SKConv

BatchNorm2d = nn.BatchNorm2d
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock

# class GDConv(nn.Module):
#     def __init__(self, in_channels):
#         super(GDConv, self).__init__()
#         # 初始化一个大小为(W×H×M)的卷积核，其中M是输入通道数
#         self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=in_channels, padding=0, stride=1)
#
#     def forward(self, x):
#         # GDConv操作，输出M个大小为1x1的特征图
#         return self.depth_conv(x).view(x.size(0), 1, 1, -1)

#SEModule 是SENet的核心，它通过自适应平均池化（nn.AdaptiveAvgPool2d）
# 和两个全连接层（实际上是1x1的卷积层 nn.Conv2d）来实现通道间的关系建模和特征重标定。
class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        # 自适应平均池化至1x1大小，以提取全局空间信息
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # self.sk_conv = SKConv(channels, M=2, r=16, stride=1, L=32)

        # 第一个1x1卷积用于降维，减少参数量和计算量
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        # 第二个1x1卷积用于恢复维度
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        # Sigmoid函数用于生成通道注意力权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        # x = self.sk_conv(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        # 通过点乘方式应用通道权重，进行特征重标定
        return module_input * x

#Bottleneck 类是一个抽象的基类，它定义了一个瓶颈块的基本结构，但没有具体实现。
# 它的子类SEBottleneck和SEResNeXtBottleneck在此基础上增加了具体的实现。
class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


#其中SEBottleneck用于SENet154，
# SEResNeXtBottleneck是SE-ResNeXt模型的瓶颈设计。它们通过增加SEModule来增强瓶颈块的特征表达能力。
class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4
    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = int(math.floor(planes * (base_width / 64)) * groups)


        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        #self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
        #                       padding=1, groups=groups, bias=False)
        self.conv2 = SKConv(in_channels=width, out_channels=width, stride=stride, M=2, r=16, L=32)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

#SENet 类定义了整个网络的结构。它包括初始的卷积层（layer0），四
# 个包含瓶颈块的层（layer1-layer4），以及最后的分类层（last_linear）。
class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]

        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,ceil_mode=True)))

        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

# 定义FaceBagNet模型A版本
def FaceBagNet_model_A(num_classes=2):
    # 创建SENet模型实例
    model = SENet(SEResNeXtBottleneck,  # 使用SEResNeXt的瓶颈结构作为基本单元
                  [2, 2, 2, 2],         # 每个阶段的瓶颈块数量，共4个阶段
                  groups=32,            # 使用32个分组卷积，有助于增加网络的容量
                  reduction=16,         # SE块中的降维比例
                  dropout_p=0.,         # 不使用dropout
                  inplanes=64,          # 初始通道数
                  input_3x3=False,      # 第一层使用标准卷积而非3x3卷积
                  downsample_kernel_size=1,  # 下采样卷积核大小
                  downsample_padding=0,      # 下采样填充
                  num_classes=num_classes)   # 最终分类的类别数
    return model

# 定义FaceBagNet模型B版本
def FaceBagNet_model_B(num_classes=2):
    # 创建SENet模型实例，与模型A相似，但在第二和第三阶段使用了更多的瓶颈块
    model = SENet(SEResNeXtBottleneck,
                  [2, 4, 4, 2],  # 第二和第三阶段增加块的数量，增强模型能力
                  groups=32,
                  reduction=16,
                  dropout_p=0.,
                  inplanes=64,
                  input_3x3=False,
                  downsample_kernel_size=1,
                  downsample_padding=0,
                  num_classes=num_classes)
    return model

# 定义FaceBagNet模型C版本
def FaceBagNet_model_C(num_classes=2):
    # 创建SENet模型实例，与模型A和B相似，但具有不同的瓶颈块配置和分组数量
    model = SENet(SEResNeXtBottleneck,
                  [3, 4, 4, 3],  # 在第一和最后阶段增加块的数量，进一步增强模型
                  groups=16,     # 减少分组数量，可能影响模型的容量和性能
                  reduction=16,
                  dropout_p=0.,
                  inplanes=64,
                  input_3x3=False,
                  downsample_kernel_size=1,
                  downsample_padding=0,
                  num_classes=num_classes)
    return model



###########################################################################################3
class Net(nn.Module):
    def __init__(self, num_class=2, is_first_bn=False, type="A"):
        """
        网络构造函数
        :param num_class: 分类任务的类别数
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
            self.encoder = FaceBagNet_model_A()
        elif type == 'B':
            self.encoder = FaceBagNet_model_B()
        elif type == 'C':
            self.encoder = FaceBagNet_model_C()
        # elif type == 'baseline':
        #     # 假设tvm.resnet18是一个函数，该函数返回一个预训练的resnet18模型
        #     self.encoder = tvm.resnet18(pretrained=False)

        # 从encoder中提取各个卷积层
        self.conv1 = self.encoder.layer0
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        # 定义一个全连接层，用于将编码器的特征映射为分类结果
        self.fc = nn.Sequential(nn.Linear(2048, num_class))

    def load_pretrain(self, pretrain_file):
        """
        加载预训练模型
        :param pretrain_file: 预训练模型文件路径
        """
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()

        # 更新当前模型的state_dict
        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key] = pretrain_state_dict['module.' + key]

        self.load_state_dict(state_dict)
        print('load: ' + pretrain_file)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入的图像数据
        """
        batch_size, C, H, W = x.shape

        # 如果使用批量归一化，则对输入数据应用批量归一化
        # 否则，对输入数据进行标准化
        if self.is_first_bn:
            x = self.first_bn(x)
        else:
            mean = [0.485, 0.456, 0.406]  # RGB通道的均值
            std = [0.229, 0.224, 0.225]  # RGB通道的标准差

            # 对每个通道的数据进行标准化处理
            x = torch.cat([
                (x[:, [0]] - mean[0]) / std[0],
                (x[:, [1]] - mean[1]) / std[1],
                (x[:, [2]] - mean[2]) / std[2],
            ], 1)

        # 数据通过各个卷积层
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # 应用自适应平均池化并调整形状
        fea = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size, -1)
        # 应用dropout
        fea = F.dropout(fea, p=0.50, training=self.training)
        # 通过全连接层得到最终的分类结果
        logit = self.fc(fea)

        return logit

    def forward_res3(self, x):
        """
        特殊的前向传播函数，只使用前三个卷积层
        :param x: 输入的图像数据
        """
        batch_size, C, H, W = x.shape

        # 和标准的前向传播一样，先进行批量归一化或标准化
        if self.is_first_bn:
            x = self.first_bn(x)
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            x = torch.cat([
                (x[:, [0]] - mean[0]) / std[0],
                (x[:, [1]] - mean[1]) / std[1],
                (x[:, [2]] - mean[2]) / std[2],
            ], 1)

        # 数据通过前三个卷积层
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x

    def set_mode(self, mode, is_freeze_bn=False):
        """
        设置模型的模式
        :param mode: 模型的模式（'eval', 'valid', 'test', 'backup'）
        :param is_freeze_bn: 是否冻结批量归一化层的参数
        """
        self.mode = mode
        # 根据模式选择模型的行为
        if mode in ['eval', 'valid', 'test']:
            # 设置为评估模式，关闭dropout等
            self.eval()
        elif mode in ['backup']:
            # 设置为训练模式
            self.train()
            # 如果is_freeze_bn为True，则冻结批量归一化层的参数
            if is_freeze_bn:
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        # 设置为评估模式，并冻结参数
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

###########################################################################################3
class FusionNet(nn.Module):

    def __init__(self, num_class=2, type='A', fusion='se_fusion'):
        """
        构造多模态融合网络 FusionNet 的构造函数
        :param num_class: 分类任务的类别数
        :param type: 使用的网络类型（'A', 'B', 'C', 'baseline'）
        :param fusion: 融合模块类型（例如 'se_fusion'）
        """
        super(FusionNet, self).__init__()

        # 初始化三个模态的网络模块：颜色(color)、深度(depth)和红外(ir)
        self.color_module = Net(num_class=num_class, is_first_bn=True, type=type)
        self.depth_module = Net(num_class=num_class, is_first_bn=True, type=type)
        self.ir_module = Net(num_class=num_class, is_first_bn=True, type=type)

        self.fusion = fusion
        # 如果融合类型是 'se_fusion'，则为每个模态初始化一个SE模块
        if fusion == 'se_fusion':
            self.color_SE = SEModule(512, reduction=16)
            self.depth_SE = SEModule(512, reduction=16)
            self.ir_SE = SEModule(512, reduction=16)

        # 定义瓶颈层，用于将三个模态的特征图并行融合
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512*3, 128*3, kernel_size=1, padding=0),
            nn.BatchNorm2d(128*3),
            nn.ReLU(inplace=True)
        )



        # 定义两个残差层，用于进一步提取特征
        self.res_0 = self._make_layer(BasicBlock, 128*3, 256, 2, stride=2)
        self.res_1 = self._make_layer(BasicBlock, 256, 512, 2, stride=2)

        # 定义全连接层，用于分类
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_class)
        )
        self.bottleneckAndfc = nn.Sequential(
            # nn.Conv2d(512, 128, kernel_size=1, padding=0),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            # self._make_layer(BasicBlock, 128, 512, 2, stride=2),
            # self._make_layer(BasicBlock, 512, 256, 2, stride=2),
            nn.AdaptiveAvgPool2d(1),  # 将特征图尺寸调整为1x1
            nn.Flatten(),  # 将特征图展平成向量
            nn.Dropout(0.5),
            nn.Linear(512, 256),  # 第一个全连接层
            nn.ReLU(inplace=True),
            nn.Linear(256, num_class),  # 最后的分类层
        )

    def load_pretrain(self, pretrain_file):
        """
        加载预训练模型的权重
        :param pretrain_file: 预训练模型文件的路径
        """
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        # 更新当前模型的 state_dict
        for key in keys:
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        """
        构建残差层
        :param block: 残差块的类型
        :param inplanes: 输入通道数
        :param planes: 输出通道数
        :param blocks: 残差块的数量
        :param stride: 步长
        """
        downsample = None
        # 如果步长不为1或者输入输出通道数不一致，需要使用downsample来匹配维度
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 添加第一个残差块（可能包含downsample）
        layers.append(block(inplanes, planes, stride, downsample))
        # 更新inplanes为扩展后的通道数
        self.inplanes = planes * block.expansion
        # 添加剩余的残差块
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward1(self, x):
        """
        前向传播函数
        :param x: 输入的多模态数据，包括颜色、深度和红外图像
        """
        batch_size, C, H, W = x.shape
        # 将输入的多模态数据分别切分为颜色、深度和红外部分
        color, depth, ir = x[:, 3:6, :, :], x[:, 0:3, :, :], x[:, 6:, :, :]
        # 通过各自的模块提取特征
        color_feas = self.color_module.forward_res3(color)
        depth_feas = self.depth_module.forward_res3(depth)
        ir_feas = self.ir_module.forward_res3(ir)

        # 如果融合类型是 'se_fusion'，则对每个模态的特征应用SE模块
        if self.fusion == 'se_fusion':
            color_feas = self.color_SE(color_feas)
            depth_feas = self.depth_SE(depth_feas)
            ir_feas = self.ir_SE(ir_feas)

        # 将三个模态的特征图拼接在一起
        fea = torch.cat([color_feas, depth_feas, ir_feas], dim=1)
        # 通过瓶颈层处理拼接后的特征图
        fea = self.bottleneck(fea)
        # x=self.bottleneck(fea)

        # 数据通过残差层
        x = self.res_0(fea)
        x = self.res_1(x)
        # 应用自适应平均池化并调整形状
        x = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size, -1)
        # 通过全连接层得到最终的分类结果
        x = self.fc(x)
        return x

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入的多模态数据，包括颜色、深度和红外图像
        """
        batch_size, C, H, W = x.shape
        # 将输入的多模态数据分别切分为颜色、深度和红外部分
        color, depth, ir = x[:, 3:6, :, :], x[:, 0:3, :, :], x[:, 6:, :, :]
        # 通过各自的模块提取特征
        color_feas = self.color_module.forward_res3(color)
        depth_feas = self.depth_module.forward_res3(depth)
        ir_feas = self.ir_module.forward_res3(ir)

        # 如果融合类型是 'se_fusion'，则对每个模态的特征应用SE模块
        if self.fusion == 'se_fusion':
            color_feas = self.color_SE(color_feas)
            depth_feas = self.depth_SE(depth_feas)
            ir_feas = self.ir_SE(ir_feas)

        # 将三个模态的特征图拼接在一起
        fea = torch.cat([color_feas, depth_feas, ir_feas], dim=1)
        # 通过瓶颈层处理拼接后的特征图
        fea = self.bottleneck(fea)

        # 数据通过残差层
        x = self.res_0(fea)
        x = self.res_1(x)
        # 应用自适应平均池化并调整形状
        x = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size, -1)
        # 通过全连接层得到最终的分类结果
        x = self.fc(x)
        integrated_prediction=nn.Sigmoid()(x)[:, 1].unsqueeze(1)

        # 获取三个模态的独立预测结果
        color_prediction = self.bottleneckAndfc(color_feas)
        color_prediction = nn.Sigmoid()(color_prediction)[:, 1].unsqueeze(1)
        

        depth_prediction = self.bottleneckAndfc(depth_feas)
        depth_prediction = nn.Sigmoid()(depth_prediction)[:, 1].unsqueeze(1)
        ir_prediction = self.bottleneckAndfc(ir_feas)
        ir_prediction = nn.Sigmoid()(ir_prediction)[:, 1].unsqueeze(1)

        return x, integrated_prediction, color_prediction, depth_prediction, ir_prediction








    def set_mode(self, mode, is_freeze_bn=False):
        """
        设置模型的模式
        :param mode: 模型的模式（'eval', 'valid', 'test', 'backup'）
        :param is_freeze_bn: 是否冻结批量归一化层的参数
        """
        self.mode = mode
        # 根据模式选择模型的行为
        if mode in ['eval', 'valid', 'test']:
            # 设置为评估模式，关闭dropout等
            self.eval()
        elif mode in ['backup']:
            # 设置为训练模式
            self.train()
            # 如果is_freeze_bn为True，则冻结批量归一化层的参数
            if is_freeze_bn:
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        # 设置为评估模式，并冻结参数
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False



# ""
# 1. 注意力机制（Attention Mechanisms）
# 注意力机制可以自动学习每个模态相对于任务的重要性。通过在多模态融合过程中引入注意力层，模型可以为不同模态分配不同的权重，从而反映出它们对当前任务的贡献大小。
# 例如，如果一个模态对于预测结果更为关键，注意力机制可以赋予这个模态更高的权重。
# 2. 可解释性技术（Explainability Techniques）
# 使用可解释性技术，如SHAP（SHapley Additive exPlanations）或LIME（Local Interpretable Model-agnostic Explanations），可以帮助揭示不同模态在模型决策过程中的作用。
# 这些方法可以为每个特征（在这种情况下是模态）分配一个重要性分数，从而提供关于各模态贡献的直观理解。
# 3. 模态消融实验（Modality Ablation Studies）
# 通过模态消融实验，可以研究移除某个模态对模型性能的影响。如果移除某个模态后性能显著下降，那么可以认为这个模态对于任务非常重要。
# 这种方法简单直接，但需要多次训练模型来比较不同模态组合的影响。
# 4. 多任务学习（Multi-Task Learning）
# 在多任务学习框架中，可以为每个模态设计一个辅助任务，并监控这些任务的学习进度。模态的学习进度和对主任务的贡献可能是相关的。
# 比如，一个模态的辅助任务如果学得很好，可能表明这个模态对于主任务也很重要。
# 5. 模态特定的网络架构（Modality-Specific Network Architectures）
# 为每个模态设计特定的网络架构，并通过模型的参数量和性能来评估每个模态的贡献。模态特定的架构可以更好地捕捉每个模态的独特特征，
# 其性能可以间接反映模态的重要性。
# ""