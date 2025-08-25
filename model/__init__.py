import torch.nn as nn


# 根据模型名称、图像大小、补丁大小和分类数来获取融合模型
def get_fusion_model(model_name, image_size, patch_size, num_class=2):
    # 如果模型名称是'MMDGFAS'
    if model_name == 'MMDGFAS':
        # 从model.MMDGFAS模块导入FusionNet类
        from model.MMDGFAS import FusionNet
        # 创建FusionNet的实例，设置分类数、类型和融合方式
        net = FusionNet(num_class=num_class, type='A', fusion='se_fusion')

    # 如果模型名称是'ViTFusion'
    elif model_name == 'ViTFusion':
        # 从model.MultiModalViT模块导入MultiModalViT类
        from model.MultiModalViT import MultiModalViT
        # 创建MultiModalViT的实例，设置各种参数
        net = MultiModalViT(img_size=image_size,
                            patch_size=patch_size,
                            in_chans=3,
                            num_classes=num_class,
                            embed_dim=384,
                            depth=6,
                            num_heads=8,
                            mlp_ratio=4.,
                            qkv_bias=False,
                            qk_scale=None,
                            drop_rate=0.2,
                            attn_drop_rate=0.1,
                            drop_path_rate=0.1,
                            norm_layer=nn.LayerNorm,
                            init_values=0.,
                            use_learnable_pos_emb=True,
                            init_scale=0.,
                            use_mean_pooling=True
                            )

    # 返回创建的网络模型实例
    return net


# 根据模型名称、图像大小、补丁大小和分类数来获取模型
def get_model(model_name, image_size, patch_size, num_class=2):
    # 如果模型名称是'MMDGFAS'
    if model_name == 'MMDGFAS':
        # 从model.MMDGFAS模块导入Net类
        from model.MMDGFAS import Net
        # 创建Net的实例，设置分类数和类型
        net = Net(num_class=num_class, type='A')

    elif model_name == 'Densenet':
        # 从model.ConvMixer模块导入ConvMixer类
        from model.Densenet import Net
        # 创建ConvMixer的实例，设置各种参数
        net = Net(type='A', num_classes=num_class)

    # 如果模型名称是'ConvMixer'
    elif model_name == 'ConvMixer':
        # 从model.ConvMixer模块导入ConvMixer类
        from model.ConvMixer import ConvMixer as Net
        # 创建ConvMixer的实例，设置各种参数
        net = Net(dim=512, depth=16, kernel_size=9, patch_size=patch_size, n_classes=num_class)

    # 如果模型名称是'MLPMixer'
    elif model_name == 'MLPMixer':
        # 从model.MLPMixer模块导入MLPMixer类
        from model.MLPMixer import MLPMixer as Net
        # 创建MLPMixer的实例，设置各种参数
        net = Net(image_size=image_size, channels=3, patch_size=patch_size, dim=512, depth=16,
                  num_classes=num_class, expansion_factor=4, dropout=0.)

    # 如果模型名称是'VisionPermutator'
    elif model_name == 'VisionPermutator':
        # 从model.ViP模块导入Permutator类
        from model.ViP import Permutator as Net
        # 创建Permutator的实例，设置各种参数
        net = Net(image_size=image_size, patch_size=patch_size, dim=512, depth=16,
                  num_classes=num_class, expansion_factor=4, segments=4, dropout=0.)

    # 如果模型名称是'ViT'
    elif model_name == 'ViT':
        # 从model.MultiModalViT模块导入MultiModalViT类
        from model.MultiModalViT import MultiModalViT
        # 创建MultiModalViT的实例，设置各种参数，注意这里设置is_multi_modal为False
        net = MultiModalViT(img_size=image_size,
                            patch_size=patch_size,
                            in_chans=3,
                            num_classes=2,
                            embed_dim=384,
                            depth=6,
                            num_heads=8,
                            mlp_ratio=4.,
                            qkv_bias=False,
                            qk_scale=None,
                            drop_rate=0.2,
                            attn_drop_rate=0.1,
                            drop_path_rate=0.1,
                            norm_layer=nn.LayerNorm,
                            init_values=0.,
                            use_learnable_pos_emb=True,
                            init_scale=0.,
                            use_mean_pooling=True,
                            is_multi_modal=False
                            )
        # 如果模型名称是'ViT'
    elif model_name == 'ViT2':
        # 从model.MultiModalViT模块导入MultiModalViT类
        from model.Vit import MultiModalViT
        # 创建MultiModalViT的实例，设置各种参数，注意这里设置is_multi_modal为False
        net = MultiModalViT(img_size=image_size,
                            patch_size=patch_size,
                            in_chans=3,
                            num_classes=2,
                            embed_dim=384,
                            depth=6,
                            num_heads=6,
                            mlp_ratio=4.,
                            qkv_bias=False,
                            qk_scale=None,
                            drop_rate=0.2,
                            attn_drop_rate=0.1,
                            drop_path_rate=0.1,
                            norm_layer=nn.LayerNorm,
                            init_values=0.,
                            use_learnable_pos_emb=True,
                            init_scale=0.,
                            use_mean_pooling=True,
                            is_multi_modal=False,
                            kernel_size=3,
                            dilation=[1, 2, 3]# Add this parameter

                            )
    elif model_name == 'ViT3':
        # 从model.MultiModalViT模块导入MultiModalViT类
        from model.fusion.MultiModalViTKan import MultiModalViT
        # 创建MultiModalViT的实例，设置各种参数，注意这里设置is_multi_modal为False
        net = MultiModalViT(img_size=image_size,
                            patch_size=patch_size,
                            in_chans=3,
                            num_classes=2,
                            embed_dim=384,
                            depth=6,
                            num_heads=8,
                            mlp_ratio=4.,
                            qkv_bias=False,
                            qk_scale=None,
                            drop_rate=0.2,
                            attn_drop_rate=0.1,
                            drop_path_rate=0.1,
                            norm_layer=nn.LayerNorm,
                            init_values=0.,
                            use_learnable_pos_emb=True,
                            init_scale=0.,
                            use_mean_pooling=True,
                            is_multi_modal=False
                            )
    # 返回创建的网络模型实例
    return net