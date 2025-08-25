import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from model.kan import KANLinear


def _cfg(url='', **kwargs):
    """
    生成和返回一个配置字典，用于模型或数据处理的参数配置。

    参数:
    - url (str): 模型权重的下载URL，默认为空字符串。
    - **kwargs: 任意数量的关键字参数，用于更新或添加到返回的配置字典中。

    返回:
    - 一个包含默认配置和通过kwargs更新或添加的配置的字典。
    """
    return {
        'url': url,  # 模型权重的下载URL。
        'num_classes': 1000,  # 模型用于分类的类别数，默认为1000。
        'input_size': (3, 224, 224),  # 模型期望的输入数据尺寸，格式为(通道数, 高度, 宽度)。
        'pool_size': None,  # 池化层的尺寸，None表示不使用池化。
        'crop_pct': .9,  # 在数据预处理时用于裁剪图像的百分比。
        'interpolation': 'bicubic',  # 图像缩放时使用的插值方法，默认为双三次插值。
        'mean': (0.5, 0.5, 0.5),  # 用于数据标准化的均值，对应于图像的三个通道。
        'std': (0.5, 0.5, 0.5),  # 用于数据标准化的标准差，对应于图像的三个通道。
        **kwargs  # 允许通过kwargs传入更多的配置项，并覆盖或添加到字典中。
    }


class DropPath(nn.Module):
    """
    DropPath 类用于实现随机深度正则化技术。

    当在残差块的主路径上应用时，它通过按样本随机丢弃路径来工作。
    """

    def __init__(self, drop_prob=None):
        """
        初始化 DropPath 模块。

        参数:
        - drop_prob (float, 可选): 路径被丢弃的概率。默认为 None，这意味着不执行路径丢弃。
        """
        super(DropPath, self).__init__()  # 调用父类的初始化方法。
        self.drop_prob = drop_prob  # 设置路径丢弃的概率。

    def forward(self, x):
        """
        定义模块的前向传播逻辑。

        参数:
        - x (Tensor): 输入张量。

        返回:
        - Tensor: 经过 DropPath 处理后的输出张量。
        """
        return drop_path(x, self.drop_prob, self.training)  # 调用 drop_path 函数处理输入张量。

    def extra_repr(self) -> str:
        """
        提供模块的额外描述字符串，用于打印和调试。

        返回:
        - str: 描述当前模块 drop_prob 属性的字符串。
        """
        return 'p={}'.format(self.drop_prob)  # 返回描述 drop_prob 属性的字符串。


class Mlp(nn.Module):
    """
    Mlp 类实现了一个基本的多层感知器（MLP）结构。

    它包含两个全连接层和一个激活层，以及一个可选的 Dropout 层。
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        初始化 Mlp 模块。

        参数:
        - in_features (int): 输入特征的数量。
        - hidden_features (int, 可选): 隐藏层的特征数量。如果没有提供，则默认为与输入特征数量相同。
        - out_features (int, 可选): 输出特征的数量。如果没有提供，则默认为与输入特征数量相同。
        - act_layer (nn.Module, 可选): 用于隐藏层的激活函数。默认为 GELU（高斯误差线性单元）。
        - drop (float, 可选): Dropout 层的概率。默认为 0，即无 Dropout。
        """
        super().__init__()  # 调用父类的初始化方法。
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features2=int(hidden_features/2)
        self.fc1 = KANLinear(in_features, hidden_features)  # 第一个全连接层。
        #self.act = act_layer()  # 激活层。

        self.fc2 = KANLinear(hidden_features, hidden_features2)  # 第二个全连接层。
        self.fc3=KANLinear(hidden_features2, out_features)
        self.drop = nn.Dropout(drop)  # Dropout 层。

    def forward(self, x):
        """
        定义模块的前向传播逻辑。

        参数:
        - x (Tensor): 输入张量。

        返回:
        - Tensor: 经过 MLP 处理后的输出张量。
        """
        x = self.fc1(x)  # 输入经过第一个全连接层。
        #x = self.act(x)  # 经过激活函数。
        # x = self.drop(x)  # 注释掉的 Dropout，原文注释提到为了保持与原始 BERT 实现一致而省略。
        x = self.fc2(x)  # 经过第二个全连接层。
        x = self.fc3(x)
        x = self.drop(x)  # 应用 Dropout。
        return x


class Attention(nn.Module):
    """
    Attention 类实现了一个自注意力机制，用于计算注意力分数并应用到输入上。
    它支持多头注意力，允许模型在不同的表示子空间并行学习。
    """
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        """
        初始化 Attention 模块。
        参数:
        - dim (int): 输入特征的维度。
        - num_heads (int): 注意力头的数量，默认为 8。
        - qkv_bias (bool): 是否为 QKV 线性变换添加偏置项，默认为 False。
        - qk_scale (float, 可选): 缩放因子，如果没有提供，则使用 head_dim ** -0.5。
        - attn_drop (float): 注意力权重的 Dropout 概率，默认为 0。
        - proj_drop (float): 输出投影的 Dropout 概率，默认为 0。
        - attn_head_dim (int, 可选): 指定注意力头的维度，默认情况下为 dim // num_heads。
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个头的维度。
        if attn_head_dim is not None:
            head_dim = attn_head_dim  # 如果提供了 attn_head_dim，使用该值作为每个头的维度。
        all_head_dim = head_dim * self.num_heads  # 所有头的总维度。
        self.scale = qk_scale or head_dim ** -0.5  # 设定缩放因子。

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)  # QKV 线性变换。
        if qkv_bias:
            # 如果启用 qkv_bias，创建和初始化偏置参数。
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)  # 注意力 Dropout 层。
        self.proj = nn.Linear(all_head_dim, dim)  # 输出投影线性层。
        self.proj_drop = nn.Dropout(proj_drop)  # 输出投影 Dropout。

    def forward(self, x):
        """
        定义模块的前向传播逻辑。
        参数:
        - x (Tensor): 输入张量，形状为 (B, N, C)。
        返回:
        - Tensor: 经过注意力机制处理后的输出张量。
        """
        B, N, C = x.shape  # 获取输入张量的批次大小、序列长度和特征维度。
        qkv_bias = None
        if self.q_bias is not None:
            # 如果 q_bias 存在，则创建 qkv_bias。
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        # 使用 F.linear 函数应用 QKV 线性变换并添加偏置。
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        # 重塑和置换得到 q, k, v。
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离 Q, K, V。

        q = q * self.scale  # 应用缩放因子。
        attn = (q @ k.transpose(-2, -1))  # 计算注意力得分。

        attn = attn.softmax(dim=-1)  # 应用 softmax 获取注意力权重。
        attn = self.attn_drop(attn)  # 应用注意力 Dropout。

        # 计算权重和值的点积，然后重塑。
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)  # 应用输出投影。
        x = self.proj_drop(x)  # 应用输出投影 Dropout。
        return x


class Block(nn.Module):
    """
    Block 类代表 Transformer 架构中的一个标准编码器块，
    包含自注意力层和 MLP 层，以及相关的归一化和 Dropout。
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., init_values=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        """
        初始化 Block 模块。

        参数:
        - dim (int): 输入特征的维度。
        - num_heads (int): 注意力头的数量。
        - mlp_ratio (float): MLP 隐藏层与输入维度的比率，默认为 4。
        - qkv_bias (bool): 是否为 QKV 线性变换添加偏置项，默认为 False。
        - qk_scale (float, 可选): 缩放因子，如果没有提供，则使用 head_dim ** -0.5。
        - drop (float): MLP 输出投影的 Dropout 概率，默认为 0。
        - attn_drop (float): 注意力权重的 Dropout 概率，默认为 0。
        - drop_path (float): DropPath 正则化的概率，默认为 0。
        - init_values (float, 可选): 初始化残差连接的缩放参数，默认为 None。
        - act_layer (nn.Module): 激活层，默认为 GELU。
        - norm_layer (nn.Module): 归一化层，默认为 LayerNorm。
        - attn_head_dim (int, 可选): 指定注意力头的维度，默认情况下为 dim // num_heads。
        """
        super().__init__()
        self.norm1 = norm_layer(dim)  # 第一个归一化层。
        self.attn = Attention(  # 自注意力层。
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # DropPath 层，如果 drop_path 为 0，则使用 Identity。
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)  # 第二个归一化层。
        mlp_hidden_dim = int(dim * mlp_ratio)  # 计算 MLP 隐藏层维度。
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)  # MLP 层。
        #self.kan=KAN()

        # 如果提供了 init_values，初始化残差连接的缩放参数（gamma）。
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        """
        定义模块的前向传播逻辑。

        参数:
        - x (Tensor): 输入张量。

        返回:
        - Tensor: 经过 Block 处理后的输出张量。
        """
        # 应用自注意力层和 MLP 层，每个层后跟一个 DropPath 层和残差连接。
        if self.gamma_1 is None:
            # 如果没有 gamma 参数，直接使用残差连接。
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            # 如果有 gamma 参数，将其应用于残差连接。
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """
    PatchEmbed 类用于将图像划分为多个小块（Patch），并将每个小块嵌入为特征向量。
    该类使用卷积操作来实现这一过程。
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """
        初始化 PatchEmbed 模块。
        参数:
        - img_size (int 或 tuple): 输入图像的尺寸。如果是整数，则表示宽和高相等。
        - patch_size (int 或 tuple): 每个小块的尺寸。如果是整数，则表示宽和高相等。
        - in_chans (int): 输入图像的通道数，默认为 3（即 RGB 图像）。
        - embed_dim (int): 每个小块嵌入的特征维度，默认为 768。
        """
        super().__init__()
        img_size = to_2tuple(img_size)  # 将图像尺寸转换为二元组 (宽, 高)。
        patch_size = to_2tuple(patch_size)  # 将小块尺寸转换为二元组 (宽, 高)。
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])  # 计算总的小块数量。
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 每个维度的小块数量。
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 使用卷积操作将图像划分为小块，并嵌入为特征向量。
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        """
        定义模块的前向传播逻辑。
        参数:
        - x (Tensor): 输入图像张量，形状为 (B, C, H, W)。
        返回:
        - Tensor: 嵌入的小块特征向量，形状为 (B, num_patches, embed_dim)。
        """
        B, C, H, W = x.shape  # 获取输入张量的批次大小、通道数、高度和宽度。
        # FIXME: 可以考虑放宽尺寸约束。
        # 确保输入图像的尺寸与初始化时设定的尺寸一致。
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 使用卷积操作将图像划分为小块，并展平和置换维度。
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class MultiModalViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224,  # 输入图像的尺寸。默认为224，符合常见的图像处理尺寸。
                 patch_size=16,  # 将图像划分成小块（patches）的尺寸。默认为16x16。
                 in_chans=3,  # 输入图像的通道数。对于RGB图像，该值为3。
                 num_classes=1000,  # 模型输出的类别数。默认为1000，适用于ImageNet等大规模数据集。
                 embed_dim=768,  # Transformer模型中隐藏层的维度。默认为768。
                 depth=12,  # Transformer模型中编码器块的数量。默认为12。
                 num_heads=12,  # 自注意力机制中头的数量。默认为12。
                 mlp_ratio=4.,  # Transformer中前馈网络的隐藏层维度与嵌入维度的比率。默认为4。
                 qkv_bias=False,  # 是否在自注意力计算中为Q（查询）、K（键）、V（值）添加偏置项。默认为False。
                 qk_scale=None,  # 自注意力中的缩放因子。如果为None，则会根据隐藏维度自动计算。
                 drop_rate=0.,  # Dropout比率，用于嵌入层和MLP层。默认为0。
                 attn_drop_rate=0.,  # 自注意力权重的Dropout比率。默认为0。
                 drop_path_rate=0.,  # 编码器块内路径的Dropout比率，用于实现Stochastic Depth。默认为0。
                 norm_layer=nn.LayerNorm,  # Transformer中使用的归一化层类型。默认为LayerNorm。
                 init_values=0.,  # 编码器块残差连接的初始化缩放值。默认为0。
                 use_learnable_pos_emb=False,  # 是否使用可学习的位置嵌入。默认为False，使用正弦余弦位置编码。
                 init_scale=0.,  # 线性层权重的初始化缩放因子。默认为0。
                 use_mean_pooling=True,  # 是否在最终特征提取时使用均值池化。默认为True。
                 is_multi_modal=True):  # 模型是否支持多模态输入。默认为True。
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.is_multi_modal = is_multi_modal
        if is_multi_modal:
            # 如果是多模态，为每种图像数据类型初始化一个 PatchEmbed 实例
            self.patch_embed0 = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            self.patch_embed1 = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            self.patch_embed2 = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            num_patches = self.patch_embed0.num_patches * 3
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 注释掉的这行代码是用于初始化一个分类（class）令牌的，这在一些Transformer模型中用于聚合序列的全局信息。在这个实现中，它暂时被注释掉了。

        if use_learnable_pos_emb:
            # 如果选择使用可学习的位置嵌入，则初始化一个全为零的位置嵌入参数。
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # 如果不使用可学习的位置嵌入，则使用正弦余弦函数生成静态的位置嵌入。
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)  # 初始化一个Dropout层，用于位置嵌入后的Dropout正则化。

        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, depth)]  # 生成一个DropPath率的列表，用于不同深度的stochastic depth正则化。
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])  # 创建一个模块列表，每个模块都是一个Transformer编码器块。

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)  # 根据是否使用均值池化选择使用Identity层还是归一化层。
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None  # 如果使用均值池化，则在最后的全连接层前添加一个额外的归一化层。

        self.head = nn.Linear(embed_dim,
                              num_classes) if num_classes > 0 else nn.Identity()  # 初始化分类头，如果类别数大于0，则使用线性层；否则使用Identity层。

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)  # 如果使用可学习的位置嵌入，用截断正态分布初始化位置嵌入的权重，标准差设为0.02。

        # trunc_normal_(self.cls_token, std=.02)  # 如果使用分类令牌（cls_token），则同样用截断正态分布初始化其权重，这里被注释掉了。

        trunc_normal_(self.head.weight, std=.02)  # 用截断正态分布初始化分类头的权重，标准差设为0.02。
        self.apply(self._init_weights)  # 调用自定义的权重初始化函数 `_init_weights` 来初始化模型中其他层的权重。

        self.head.weight.data.mul_(init_scale)  # 将分类头的权重乘以初始化缩放因子 `init_scale`。
        self.head.bias.data.mul_(init_scale)  # 将分类头的偏置也乘以初始化缩放因子 `init_scale`。

    def _init_weights(self, m):
        # 检查传入的模块 m 是否是 nn.Linear 类型（全连接层）
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)  # 如果是，使用截断正态分布初始化它的权重，标准差设为0.02。
            # 进一步检查全连接层的偏置是否存在
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 如果存在，将偏置初始化为0。

        # 检查传入的模块 m 是否是 nn.LayerNorm 类型（层归一化）
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  # 将层归一化的偏置初始化为0。
            nn.init.constant_(m.weight, 1.0)  # 将层归一化的权重初始化为1。

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):

        if self.is_multi_modal:
            color, depth, ir = x[:, 3:6, :, :], x[:, 0:3, :, :], x[:, 6:, :, :]
            x0 = self.patch_embed0(color)
            x1 = self.patch_embed1(depth)
            x2 = self.patch_embed2(ir)
            x = torch.cat([x0,x1,x2], dim=1)
        else:
            x = self.patch_embed(x)

        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.fc_norm is not None:
            t = x[:, 1:, :]
            return self.fc_norm(t.mean(1))
        else:
            return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

if __name__ == '__main__':
    model = MultiModalViT(img_size = 96, patch_size = 8, in_chans = 3, num_classes = 2)
    print(model)
    input0 = torch.zeros([1,3,96,96])
    input1 = torch.zeros([1,3,96,96])
    input2 = torch.zeros([1,3,96,96])
    input = torch.cat([input0,input1,input2], dim=1)
    out = model(input)
    print(out)