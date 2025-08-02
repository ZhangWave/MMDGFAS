import cv2
# 引入自定义的数据帮助模块和工具模块0
from process.data_helper_padisi import *
from utils import *
RESIZE_SIZE = 112
class FDDataset_padisi_R(Dataset):
    # 类的初始化方法
    def __init__(self, mode, modality='color', fold_index=-1, image_size=128, augment = None, augmentor = None, balance = True):
        super(FDDataset_padisi_R, self).__init__()
        # 打印当前折叠索引和模态类型
        print('fold: '+str(fold_index))
        print(modality)

        # 初始化类变量
        self.mode = mode  # 数据集模式（训练、验证、测试）
        self.modality = modality  # 模态（颜色、深度、红外）

        # 数据增强函数和工具
        self.augment = augment
        self.augmentor = augmentor
        self.balance = balance  # 是否平衡数据集

        self.channels = 3  # 图像通道数
        self.train_image_path = TRN_IMGS_DIR  # 训练图像路径
        self.test_image_path = TST_IMGS_DIR  # 测试图像路径
        self.image_size = image_size  # 图像大小
        self.fold_index = fold_index  # 折叠索引

        # 设置数据集模式
        self.set_mode(self.mode,self.fold_index)

    # 设置数据集模式的方法
    def set_mode(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index
        # 打印折叠索引设置信息
        print('fold index set: ', fold_index)

        # 根据模式加载相应的列表并打印状态
        if self.mode == 'test':
            self.test_list = load_test_list()
            self.num_data = len(self.test_list)
            print('set dataset mode: test')
        elif self.mode == 'val':
            ##['Val/0000/000000-color.jpg', 'Val/0000/000000-depth.jpg', 'Val/0000/000000-ir.jpg', '0']
            self.val_list = load_val_list()
            self.num_data = len(self.val_list)
            print('set dataset mode: test')
        elif self.mode == 'train':
            self.train_list = load_train_list()
            random.shuffle(self.train_list)
            self.num_data = len(self.train_list)
            if self.balance:
                self.train_list = transform_balance(self.train_list)
            print('set dataset mode: train')
        print(self.num_data)

    # 获取数据集中一个项目的方法
    def __getitem__(self, index):
        # 检查fold_index是否为空，这是用来确保我们有一个有效的数据集的分割索引
        if self.fold_index is None:
            print('WRONG!!!!!!! fold index is NONE!!!!!!!!!!!!!!!!!')
            return

        # 准备数据项，根据数据集的模式（训练、验证或测试）
        if self.mode == 'train':
            # 如果数据集需要平衡，则根据随机选择使用正样本列表或负样本列表
            if self.balance:
                # 随机选择使用正样本列表或负样本列表
                tmp_list = self.train_list[random.randint(0, 1)]

                # 从选定的列表中随机选择一个样本
                pos = random.randint(0, len(tmp_list) - 1)
                color, depth, ir, label = tmp_list[pos]
            else:
                # 如果不需要平衡，则直接使用索引从训练列表中获取样本
                color, depth, ir, label = self.train_list[index]
        elif self.mode == 'val':
            # 直接使用索引从验证列表中获取样本
            color,depth,ir,label = self.val_list[index]
        elif self.mode == 'test':
            # 直接使用索引从测试列表中获取样本，测试模式不涉及标签
            color, depth, ir = self.test_list[index]
            # 构造一个唯一的测试ID，用于标识测试样本
            test_id = color + ' ' + depth + ' ' + ir


        # 根据modality（颜色、深度或红外）选择正确的图像路径
        if self.modality == 'color':
            img_path = os.path.join(DATA_ROOT, color)
            # print(img_path)
        elif self.modality == 'depth':
            img_path = os.path.join(DATA_ROOT, depth)
        elif self.modality == 'ir':
            img_path = os.path.join(DATA_ROOT, ir)

        # 使用OpenCV加载图像
        image = cv2.imread(img_path, 1)
        # print(image.shape)
        # 将图像调整为指定的大小
        image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))
        

        # 根据当前模式对图像进行相应的处理
        if self.mode == 'train':
            # 应用数据增强，然后调整图像大小和格式
            image = self.augment(image, target_shape=(self.image_size, self.image_size, 3))
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([self.channels, self.image_size, self.image_size])
            image = image / 255.0  # 归一化
            label = int(label)
            # 返回处理后的图像和标签
            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))
        elif self.mode == 'val':
            # 对于验证模式，进行类似的处理，但可能会使用不同的数据增强方式
            image = self.augment(image, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            n = len(image)
            image = np.concatenate(image, axis=0)
            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, self.channels, self.image_size, self.image_size])
            image = image / 255.0  # 归一化
            label = int(label)
            # 返回处理后的图像和标签
            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))
        elif self.mode == 'test':
            # 对于测试模式，只返回处理后的图像和测试ID，不涉及标签
            image = self.augment(image, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            n = len(image)
            image = np.concatenate(image, axis=0)
            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, self.channels, self.image_size, self.image_size])
            image = image / 255.0  # 归一化
            return torch.FloatTensor(image), test_id


    def __len__(self):
        return self.num_data


# check #################################################################
# 验证训练数据的函数
def run_check_train_data():
    from augmentation import color_augumentor
    augment = color_augumentor
    dataset = FDDataset_padisi_R(mode = 'val', fold_index=-1, image_size=32,  augment=augment)
    print(dataset)

    num = len(dataset)
    for m in range(num):
        i = np.random.choice(num)
        image, label = dataset[m]
        print(image.shape)
        print(label.shape)

# 主函数
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    run_check_train_data()


