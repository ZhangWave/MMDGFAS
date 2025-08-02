import math
import cv2
from imgaug import augmenters as iaa
from process.data_helper import *
from process.data_helper_wmca import *
from PIL import Image
from help.AugMix import AugMix
from torchvision.transforms import InterpolationMode
def get_augment(image_mode):
    if image_mode == 'color':
        augment = color_augumentor
    elif image_mode == 'depth':
        augment = depth_augumentor
    elif image_mode == 'ir':
        augment = ir_augumentor
    return augment

def random_cropping(image, target_shape=(32, 32, 3), is_random = True):
    image = cv2.resize(image,(RESIZE_SIZE,RESIZE_SIZE))
    target_h, target_w,_ = target_shape
    height, width, _ = image.shape

    if is_random:
        start_x = random.randint(0, width - target_w)
        start_y = random.randint(0, height - target_h)
    else:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:]
    return zeros

def TTA_5_cropps(image, target_shape=(32, 32, 3)):
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    width, height, d = image.shape
    target_w, target_h, d = target_shape

    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2

    starts = [[start_x, start_y],
              [start_x - target_w, start_y],
              [start_x, start_y - target_w],
              [start_x + target_w, start_y],
              [start_x, start_y + target_w],]

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
        images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))

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

def TTA_36_cropps(image, target_shape=(32, 32, 3)):
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
        image_flip_lr = zeros.copy()

        zeros = np.flipud(zeros)
        image_flip_lr_up = zeros.copy()

        zeros = np.fliplr(zeros)
        image_flip_up = zeros.copy()

        images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_lr.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_up.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_lr_up.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))

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

