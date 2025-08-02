from imgaug import augmenters as iaa
import math
import cv2
import random
import numpy as np

from PIL import Image
from imgaug import augmenters as iaa
from torchvision.transforms import InterpolationMode
from help.AugMix import AugMix
random.seed(555)

RESIZE_SIZE = 112


def random_cropping(image, target_shape=(32, 32, 3), is_random=True):
    """
    对输入图像进行随机裁剪或中心裁剪，以达到目标形状大小。

    参数:
    image: 输入的原始图像数组。
    target_shape: 目标图像的形状，以tuple形式表示，例如(32, 32, 3)。
                  默认为(32, 32, 3)，表示裁剪后的图像大小为32x32，颜色通道为3。
    is_random: 是否进行随机裁剪。如果为True，则从原始图像中随机选择一个矩形区域进行裁剪；
               如果为False，则从原始图像的中心区域进行裁剪。默认为True。

    返回:
    zeros: 裁剪后的图像数组。
    """

    # 首先，将输入图像调整到一个预定义的大小（RESIZE_SIZE变量需要在函数外部定义）。
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    # 获取目标裁剪图像的高度和宽度。
    target_h, target_w, _ = target_shape

    # 获取调整大小后的图像的高度和宽度。
    height, width, _ = image.shape

    # 如果选择进行随机裁剪：
    if is_random:
        # 在宽度范围内随机选择一个起始点。
        start_x = random.randint(0, width - target_w)
        # 在高度范围内随机选择一个起始点。
        start_y = random.randint(0, height - target_h)
    # 如果不进行随机裁剪，则进行中心裁剪：
    else:
        # 计算宽度的中心起始点。
        start_x = (width - target_w) // 2
        # 计算高度的中心起始点。
        start_y = (height - target_h) // 2

    # 从原始图像中裁剪出目标区域。
    cropped_image = image[start_y:start_y + target_h, start_x:start_x + target_w, :]

    # 返回裁剪后的图像。
    return cropped_image

import cv2

def TTA_5_cropps(image, target_shape=(32, 32, 3)):
    """
    从输入图像中创建五个不同的裁剪（中心，四个角落），通常用于测试时增强（TTA）。

    参数:
    image: 输入的原始图像数组。
    target_shape: 目标图像的形状，以tuple形式表示，例如(32, 32, 3)。
                  默认为(32, 32, 3)，表示裁剪后的图像大小为32x32，颜色通道为3。

    返回:
    images: 包含五个裁剪图像的列表。
    """

    # 首先，将输入图像调整到一个预定义的大小（RESIZE_SIZE变量需要在函数外部定义）。
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    # 获取调整大小后图像的宽度、高度。
    width, height, _ = image.shape
    # 获取目标裁剪图像的宽度、高度。
    target_w, target_h, _ = target_shape

    # 计算中心裁剪的起始点。
    start_x = (width - target_w) // 2
    start_y = (height - target_h) // 2

    # 定义五个裁剪的起始点：中心，四个角落。
    starts = [
        [start_x, start_y],  # 中心
        [start_x - target_w, start_y],  # 左上角
        [start_x, start_y - target_h],  # 右上角
        [start_x + target_w, start_y],  # 左下角
        [start_x, start_y + target_h],  # 右下角
    ]

    images = []  # 用于存储裁剪后的图像

    # 遍历每个起始点进行裁剪。
    for start_index in starts:
        image_ = image.copy()  # 复制原始图像
        x, y = start_index

        # 检查起始点坐标，确保不超出图像边界。
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w - 1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h - 1

        # 根据当前起始点裁剪图像。
        cropped_image = image_[y: y + target_h, x: x + target_w, :]
        images.append(cropped_image.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))

    # 返回包含五个裁剪图像的列表。
    return images

def TTA_18_cropps(image, target_shape=(32, 32, 3)):
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    width, height, d = image.shape
    target_w, target_h, d = target_shape

    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2

    starts = [[start_x, start_y],

              [start_x - target_w, start_y],
              [start_x, start_y - target_w],
              [start_x + target_w, start_y],
              [start_x, start_y + target_w],

              [start_x + target_w, start_y + target_w],
              [start_x - target_w, start_y - target_w],
              [start_x - target_w, start_y + target_w],
              [start_x + target_w, start_y - target_w],
              ]

    images = []

    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w-1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h-1

        zeros = image_[x:x + target_w, y: y+target_h, :]

        image_ = zeros.copy()
        zeros = np.fliplr(zeros)
        image_flip = zeros.copy()

        images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))

    return images
import cv2
import numpy as np

def TTA_36_cropps(image, target_shape=(32, 32, 3)):
    """
    从输入图像中创建36个不同的裁剪版本，每个版本都有4种翻转变换（无翻转、水平翻转、
    垂直翻转、水平+垂直翻转），用于测试时增强（TTA）。

    参数:
    image: 输入的原始图像数组。
    target_shape: 目标图像的形状，以tuple形式表示，例如(32, 32, 3)。

    返回:
    images: 包含36个裁剪并翻转的图像的列表。
    """

    # 将输入图像调整到一个预定义的大小。
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    # 获取调整后图像的尺寸。
    width, height, _ = image.shape
    target_w, target_h, _ = target_shape

    # 计算中心裁剪的起始点。
    start_x = (width - target_w) // 2
    start_y = (height - target_h) // 2

    # 定义初始的九个裁剪起始点：中心，四周和四个角落。
    starts = [
        [start_x, start_y],  # 中心
        [start_x - target_w, start_y],  # 左
        [start_x, start_y - target_h],  # 上
        [start_x + target_w, start_y],  # 右
        [start_x, start_y + target_h],  # 下
        [start_x + target_w, start_y + target_h],  # 右下角
        [start_x - target_w, start_y - target_h],  # 左上角
        [start_x - target_w, start_y + target_h],  # 左下角
        [start_x + target_w, start_y - target_h],  # 右上角
    ]

    images = []  # 用于存储裁剪后的图像

    # 对每个起始点进行裁剪和翻转操作。
    for start_index in starts:
        image_ = image.copy()  # 复制原始图像进行操作
        x, y = start_index

        # 确保裁剪不会超出图像边界。
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w - 1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h - 1

        # 裁剪图像。
        cropped = image_[y: y + target_h, x: x + target_w, :]

        # 生成四种翻转变换的图像。
        image_flip_lr = np.fliplr(cropped)  # 水平翻转
        image_flip_ud = np.flipud(cropped)  # 垂直翻转
        image_flip_lr_ud = np.fliplr(image_flip_ud)  # 水平+垂直翻转

        # 将原始裁剪及其翻转变换添加到列表中。
        images.append(cropped.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))
        images.append(image_flip_lr.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))
        images.append(image_flip_ud.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))
        images.append(image_flip_lr_ud.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))

    # 返回包含36个裁剪并翻转的图像的列表。
    return images

def random_erasing(img, probability = 0.5, sl = 0.02, sh = 0.5, r1 = 0.5, channel = 3):
    if random.uniform(0, 1) > probability:
        return img

    for attempt in range(100):
        area = img.shape[0] * img.shape[1]

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)

            noise = np.random.random((h,w,channel))*255
            noise = noise.astype(np.uint8)

            if img.shape[2] == channel:
                img[x1:x1 + h, y1:y1 + w, :] = noise
            else:
                print('wrong')
                return
            return img

    return img

def random_resize(img, probability = 0.5,  minRatio = 0.2):
    if random.uniform(0, 1) > probability:
        return img

    ratio = random.uniform(minRatio, 1.0)

    h = img.shape[0]
    w = img.shape[1]

    new_h = int(h*ratio)
    new_w = int(w*ratio)

    img = cv2.resize(img, (new_w,new_h))
    img = cv2.resize(img, (w, h))
    return img

def color_augumentor(image, target_shape=(32, 32, 3), is_infer=False):
    if is_infer:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0),
        ])

        image = augment_img.augment_image(image)
        image = TTA_36_cropps(image, target_shape)
        return image

    else:
        # random_value = np.random.randint(0, 2)  # 生成随机数0或1
        random_value=1

        if random_value == 0:
            # 使用第一个数据增强方法
            augment_img = iaa.Sequential([
                iaa.Fliplr(0.5),  # 50%的概率进行水平翻转
                iaa.Flipud(0.5),  # 50%的概率进行垂直翻转
                iaa.Affine(rotate=(-30, 30)),  # 在-30度到30度范围内随机旋转图像
            ], random_order=True)  # 随机顺序应用增强操作

            # 对图像进行数据增强
            image = augment_img.augment_image(image)  # 应用数据增强操作
            image = random_resize(image)  # 调整图像大小
            image = random_cropping(image, target_shape, is_random=True)  # 随机裁剪图像
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换图像颜色通道顺序
            image_pil = Image.fromarray(image_rgb)  # 转换为PIL图像对象

            # 实例化AugMix对象
            augmix = AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0, all_ops=True,
                            interpolation=InterpolationMode.BILINEAR, fill=None)

            # 调用forward方法进行数据增强
            image = augmix(image_pil)  # 使用AugMix数据增强
            # 将PIL图像对象转换为NumPy数组
            image_np = np.array(image)
            image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # 转换图像颜色通道顺序
            image = random_cropping(image, target_shape, is_random=True)  # 随机裁剪图像

        # 返回增强后的图像
        return image

def depth_augumentor(image, target_shape=(32, 32, 3), is_infer=False):
    if is_infer:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0),
        ])

        image =  augment_img.augment_image(image)
        image = TTA_36_cropps(image, target_shape)
        return image

    else:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
        ], random_order=True)

        image = augment_img.augment_image(image)
        image = random_resize(image)
        image = random_cropping(image, target_shape, is_random=True)
        return image

def ir_augumentor(image, target_shape=(32, 32, 3), is_infer=False):
    if is_infer:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0),
        ])
        image =  augment_img.augment_image(image)
        image = TTA_36_cropps(image, target_shape)
        return image

    else:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
        ], random_order=True)

        image = augment_img.augment_image(image)
        image = random_resize(image)
        image = random_cropping(image, target_shape, is_random=True)
        return image

