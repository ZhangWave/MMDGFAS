'''

Pytorch code for 'Chalearn Multi-modal Cross-ethnicity Face anti-spoofing Recognition Challenge@CVPR2020'
By Qing Yang, 2020/03/01

MIT License
Copyright (c) 2019

'''

import os


DATA_ROOT = r'/root/autodl-tmp/CASIA-CeFA/'
DATA_ROOT1 = DATA_ROOT + 'phase1'
DATA_ROOT2 = DATA_ROOT + 'phase2'
# E:\code\newAi\code\pipenet-pytorch-master\pipenet-pytorch-master\dataset\phase1
DATA_LIST1 = r'/root/code/pipenet/pipenet/dataset/phase1'
DATA_LIST2 = r'/root/code/pipenet/pipenet/dataset/phase2'
# DATA_ROOT3 = r'./dataset/phase1'

DATA_ROOT = {'train': DATA_ROOT1, 'val': DATA_ROOT1, 'dev': DATA_ROOT1, 'test': DATA_ROOT2}
DATA_LIST = {'train': DATA_LIST1, 'val': DATA_LIST1, 'dev': DATA_LIST1, 'test': DATA_LIST2}

RESIZE_SIZE = 112


def load_list(file_dir):
    list = []
    f = open(file_dir)
    lines = f.readlines()
    return lines


def write_list(ori_list, res_list, name):
    file = open(name, "w+")
    i = 0
    print("len(ori_list):", len(ori_list))
    print("len(res_list):", len(res_list))
    for line in ori_list:
        tmp = line.replace('\n', '') + " " + res_list[i] + "\n"
        file.write(str(tmp))
        i += 1
    file.close()


def load_train_list(dataset_name):
    list = []
    # f = open(DATA_ROOT + '/train_list.txt')
    f = open(DATA_LIST1 + '/' + dataset_name + '_new_train.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    # print("train_list", list)
    return list


def load_val_list(dataset_name):
    list = []
    f = open(DATA_LIST1 + '/' + dataset_name + '_new_val.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    # print("val_list",list)
    return list


def load_test_list(mode, dataset_name):
    list = []
    f = open(DATA_LIST[mode] + '/' + dataset_name + '_new_' + mode + '.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)

    return list


def transform_balance(train_list):
    print('balance!')
    pos_list = []
    neg_list = []
    for tmp in train_list:
        if tmp[3] == '1':
            pos_list.append(tmp)
        else:
            neg_list.append(tmp)

    print(len(pos_list))
    print(len(neg_list))
    return [pos_list, neg_list]

# main #################################################################
if __name__ == '__main__':
    import os
    dataset_name='4@1'
    train_list = load_train_list(dataset_name)
    transform_balance(train_list)

    # 加载测试列表

    load_val_list = load_val_list(dataset_name)
    load_train_list = load_train_list(dataset_name)

    # 打印前10项
    for item in load_val_list[:10]:
        print(item)
    # 打印前10项
    for item in load_train_list[:10]:
        print(item)
