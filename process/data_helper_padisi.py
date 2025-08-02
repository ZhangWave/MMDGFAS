import os
import random
from utils import *

# 设置数据根目录
DATA_ROOT =  r'/root/autodl-tmp/padisi_face/'
# 设置训练和测试图像的目录
TRN_IMGS_DIR = DATA_ROOT
TST_IMGS_DIR = DATA_ROOT
# 设置图片调整大小后的大小
RESIZE_SIZE = 112

def load_train_list():
    list = []
    f = open(DATA_ROOT + '/train.txt')
    lines = f.readlines()
    f.close()

    for line in lines:
        # 移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
        line = line.strip().split(' ')
        list.append(line)
    return list
# 加载验证列表
def load_val_list():
    list = []
    f = open(DATA_ROOT + '/val.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
        # ['Val/0000/000000-color.jpg', 'Val/0000/000000-depth.jpg', 'Val/0000/000000-ir.jpg', '0']
    return list

# 加载测试列表
def load_test_list():
    list = []
    f = open(DATA_ROOT + '/test.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)

    return list

# 对训练列表进行平衡处理
def transform_balance(train_list):
    pos_list = []
    neg_list = []
    for tmp in train_list:
        # 根据标签将数据分为正负两类
        if tmp[3]=='1':
            pos_list.append(tmp)
        else:
            neg_list.append(tmp)

    # 打印正负样本数量
    print(len(pos_list))
    print(len(neg_list))
    return [pos_list,neg_list]

# # 提交结果
# def submission(probs, outname, mode='valid'):
#     # 根据模式选择相应的列表文件
#     if mode == 'valid':
#         f = open(DATA_ROOT + 'val.txt')
#     else:
#         f = open(DATA_ROOT + 'test.txt')
#
#     lines = f.readlines()
#     f.close()
#     lines = [tmp.strip() for tmp in lines]
#
#     # 写入预测结果到指定文件
#     f = open(outname,'w')
#     for line,prob in zip(lines, probs):
#         out = line + ' ' + str(prob)
#         f.write(out+'\n')
#     f.close()
#     return list

def submission(probs, outname, mode='valid'):
    # 根据模式选择相应的列表文件
    if mode == 'valid':
        f = open(DATA_ROOT + 'val.txt')
    else:
        f = open(DATA_ROOT + 'test.txt')

    lines = f.readlines()
    f.close()
    lines = [tmp.strip() for tmp in lines]

    # 初始化一个新的列表来存储输出行
    output_lines = []

    # 写入预测结果到指定文件
    f = open(outname, 'w')
    for line, prob in zip(lines, probs):
        # 创建一个包含样本标识符和概率的字符串
        out = line + ' ' + str(prob)
        # 将字符串写入文件，并在每个预测后添加换行符
        f.write(out + '\n')
    # 关闭输出文件
    f.close()

    return output_lines  # 返回包含修改后的行的列表

# 主函数
if __name__ == '__main__':
    # 加载测试列表
    # submission()

    load_val_list=load_val_list()
    load_train_list=load_train_list()
    load_test_list=load_test_list()

        # 打印前10项
    for item in load_val_list[:10]:
        print(item)
    # 打印前10项
    for item in load_train_list[:10]:
        print(item)

    for item in load_test_list[:10]:
        print(item)
