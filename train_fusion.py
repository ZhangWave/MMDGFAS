import os
import sys
import argparse

from torch.utils.data import ConcatDataset

from process.data_fusion import FDDataset_sufr
from process.data_fusion_cefa import FDDataset_cefa
from process.data_fusion_wmca import FDDataset_wmca
from process.data_fusion_padisi import FDDataset_padisi
from metric import metric, do_valid_test, infer_test
from model import get_fusion_model, get_model
from loss.cyclic_lr import CosineAnnealingLR_with_Restart
from process.data_helper import submission
from utils import *
from cmfl import CMFL
import os
import numpy as np
import torch



def create_datasets(train_test):
    if train_test == 'SWP2C':
        # 使用sufr和wmca进行训练
        train_dataset1 = FDDataset_sufr(mode='train', image_size=config.image_size,
                                        fold_index=config.train_fold_index)
        train_dataset2 = FDDataset_padisi(mode='train', image_size=config.image_size,
                                        fold_index=config.train_fold_index)

        train_dataset3 = FDDataset_wmca(mode='train', image_size=config.image_size,
                                        fold_index=config.train_fold_index)
        train_dataset = ConcatDataset([train_dataset1,train_dataset2, train_dataset3])

        # 使用cefa进行测试
        valid_dataset = FDDataset_cefa(mode='val', image_size=config.image_size, dataset_name='4@1',
                                       fold_index=config.train_fold_index)
    elif train_test == 'SCP2W':
        # 使用sufr和cefa进行训练
        train_dataset1 = FDDataset_sufr(mode='train', image_size=config.image_size,
                                        fold_index=config.train_fold_index)
        train_dataset2 = FDDataset_cefa(mode='train', image_size=config.image_size, dataset_name='4@1',
                                        fold_index=config.train_fold_index)
        train_dataset3 = FDDataset_padisi(mode='train', image_size=config.image_size,
                                        fold_index=config.train_fold_index)

        # 初始化空列表用于存储多个FDDataset_padisi数据集
        padisi_datasets = []

        # 循环生成多个FDDataset_padisi数据集
        num_padisi_datasets = 9  # 指定生成的padisi数据集数量
        for i in range(num_padisi_datasets):
            padisi_dataset = FDDataset_padisi(mode='train', image_size=config.image_size,
                                              fold_index=config.train_fold_index)
            padisi_datasets.append(padisi_dataset)
        padisi_concat_dataset = ConcatDataset(padisi_datasets)

        train_dataset = ConcatDataset([train_dataset1, train_dataset2,train_dataset3,padisi_concat_dataset])

        # 使用wmca进行测试
        valid_dataset = FDDataset_wmca(mode='val', image_size=config.image_size,
                                       fold_index=config.train_fold_index)

    elif train_test == 'SCW2P':
        # 使用sufr和cefa进行训练
        train_dataset1 = FDDataset_sufr(mode='train', image_size=config.image_size,
                                        fold_index=config.train_fold_index)
        train_dataset2 = FDDataset_cefa(mode='train', image_size=config.image_size, dataset_name='4@1',
                                        fold_index=config.train_fold_index)
        train_dataset3 = FDDataset_wmca(mode='train', image_size=config.image_size,
                                        fold_index=config.train_fold_index)
        train_dataset = ConcatDataset([train_dataset1, train_dataset2, train_dataset3])

        # 使用wmca进行测试
        valid_dataset = FDDataset_padisi(mode='val', image_size=config.image_size,
                                         fold_index=config.train_fold_index)
    elif train_test == 'P2P':
        # 使用sufr和cefa进行训练
        train_dataset1 = FDDataset_padisi(mode='train', image_size=config.image_size,
                                        fold_index=config.train_fold_index)
        train_dataset = ConcatDataset([train_dataset1])

        # 使用wmca进行测试
        valid_dataset = FDDataset_padisi(mode='val', image_size=config.image_size,
                                         fold_index=config.train_fold_index)
    # 其他情况可以继续添加elif分支

    return train_dataset, valid_dataset

def run_train(config):
    model_name = f'{config.model}_{config.image_mode}_{config.image_size}_{config.train_test}'
    if 'FaceBagNet' not in config.model:
        model_name += f'_{config.patch_size}'
    config.save_dir = os.path.join(config.save_dir, model_name)

    initial_checkpoint = config.pretrained_model
    # criterion          = softmax_cross_entropy_criterion
    criterion=binary_cross_entropy_criterion


    # criterion_cmfl = CMFL(alpha=1, gamma= 3, binary= False, multiplier=2)

    ## setup  -----------------------------------------------------------------------------
    if not os.path.exists(config.save_dir +'/checkpoint'):
        os.makedirs(config.save_dir +'/checkpoint')
    if not os.path.exists(config.save_dir +'/backup'):
        os.makedirs(config.save_dir +'/backup')
    if not os.path.exists(config.save_dir +'/backup'):
        os.makedirs(config.save_dir +'/backup')

    log = Logger()
    log.open(os.path.join(config.save_dir,model_name+'.txt'),mode='a')
    log.write('\tconfig.save_dir      = %s\n' % config.save_dir)
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... xxx baseline  ... \n')
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')


    train_dataset, valid_dataset = create_datasets(config.train_test)
    train_loader  = DataLoader(train_dataset,
                                shuffle=True,
                                batch_size  = config.batch_size,
                                drop_last   = True,
                                num_workers = config.num_workers)

    # valid_dataset = FDDataset_wmca(mode = 'val', image_size=config.image_size,
    #                           fold_index=config.train_fold_index)

    print("Total number of samples in valid_dataset:", len(valid_dataset))

    valid_loader  = DataLoader(valid_dataset,
                                shuffle=False,
                                batch_size  = config.batch_size // 36,
                                drop_last   = False,
                                num_workers = config.num_workers)

    assert(len(train_dataset)>=config.batch_size)
    log.write('batch_size = %d\n'%(config.batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')
    log.write('** net setting **\n')

    net = get_fusion_model(model_name=config.model, image_size=config.image_size, patch_size=config.patch_size)
    # print(net)
    net = torch.nn.DataParallel(net)
    net =  net.cuda()

    # 检查是否有初始检查点（即预训练的模型权重）需要加载
    if initial_checkpoint is not None:
        # 构建初始检查点的完整路径
        initial_checkpoint = os.path.join(config.save_dir + '/checkpoint', initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)  # 打印初始检查点的路径
        # 加载初始检查点的模型权重
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    # 将模型类型记录到日志文件中
    log.write('%s\n' % (type(net)))
    log.write('\n')

    # 设置平滑迭代次数。这个参数用于控制日志输出的平滑度，例如，在每20次迭代后打印一次平均损失
    iter_smooth = 20
    # 设置起始迭代次数。通常用于继续之前中断的训练过程
    start_iter = 0
    log.write('\n')  # 在日志文件中添加一个空行，用于分隔不同部分的内容

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('                                  |------------ VALID -------------         |-------- TRAIN/BATCH ----------|         \n')
    log.write('model_name   lr   iter  epoch     |     loss      acer      acc     hter    |     loss              acc     |  time   \n')
    log.write('----------------------------------------------------------------------------------------------------\n')

    train_loss   = np.zeros(6,np.float32)
    valid_loss   = np.zeros(6,np.float32)
    batch_loss   = np.zeros(6,np.float32)
    iter = 0
    i    = 0

    start = timer()
    #-----------------------------------------------
    # 初始化一个SGD优化器，用于调整模型参数
    # 这里使用filter函数来选择那些需要梯度计算（即可训练）的参数
    # 设置学习率(lr)为0.1，动量(momentum)为0.9，权重衰减(weight_decay)为0.0005用于正则化
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.1, momentum=0.9, weight_decay=0.0005)

    # 使用余弦退火（Cosine Annealing）策略来调整学习率，并设置重启策略
    # T_max参数指定了每次学习率周期的迭代次数，这里从配置中获取
    # T_mult设置为1表示每次重启后周期长度不变
    # take_snapshot设置为False，表示不在每次重启时保存模型快照
    # out_dir设置为None，表示不输出日志到目录
    # eta_min设置为1e-3，表示学习率下降的最小值
    sgdr = CosineAnnealingLR_with_Restart(optimizer,
                                          T_max=config.cycle_inter,
                                          T_mult=1,
                                          model=net,
                                          take_snapshot=False,
                                          out_dir=None,
                                          eta_min=1e-3)

    # 初始化全局最小ACER（活体检测错误率）值为1.0，用于在训练过程中追踪ACER的最小值
    global_min_acer = 1.0
    global_max_acc = 0

    # 对于配置中指定的循环次数，进行多次的训练循环
    for cycle_index in range(config.cycle_num):
        print('cycle index: ' + str(cycle_index))  # 打印当前的循环索引
        min_acer = 1.0  # 初始化当前循环的最小活体检测错误率ACER
        max_acc=0.0

        # 对于每个循环内的迭代次数，执行训练和验证
        for epoch in range(0, config.cycle_inter):
            sgdr.step()  # 更新学习率调整器的状态
            lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
            print('lr : {:.4f}'.format(lr))  # 打印当前学习率

            sum_train_loss = np.zeros(6, np.float32)  # 初始化累计训练损失
            sum = 0  # 初始化损失和的计数器
            optimizer.zero_grad()  # 清除之前的梯度

            # 迭代训练数据加载器中的每个批次
            for input, truth in train_loader:
                iter = i + start_iter  # 更新当前的总迭代数
                net.train()  # 设置网络为训练模式
                input = input.cuda()  # 将输入数据移动到GPU
                truth = truth.cuda()  # 将真值标签移动到GPU

                print('input.shape',input.shape)


                # logit,a_logit,c_logit,d_logit,i_logit = net.forward(input)  # 前向传播

                logit= net.forward(input)  # 前向传播

                print("logit shape:", logit.shape)
                print("logit values (first 5):", logit[:5])

                target = truth  #torch.Size([128, 1])

                truth = truth.view(logit.shape[0])  # 调整真值标签的形状 torch.Size([128])



                loss = criterion(logit, truth)  # 计算损失
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # loss=compute_loss(a_logit,c_logit,d_logit,i_logit,target,device)
                precision, _ = metric(logit, truth)  # 计算精度

                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重
                optimizer.zero_grad()  # 再次清除梯度

                # 记录批次的损失和精度
                batch_loss[:2] = np.array((loss.item(), precision.item(),))
                sum += 1  # 累加损失和的计数器
                if iter % iter_smooth == 0:
                    train_loss = sum_train_loss / sum  # 计算平均训练损失
                    sum = 0  # 重置损失和的计数器

                i += 1  # 更新批次计数

            # 如果达到循环迭代的一半，开始执行验证过程
            if epoch >= config.cycle_inter // 2:
                net.eval()  # 设置网络为验证模式
                valid_loss, _ = do_valid_test(net, valid_loader, criterion)  # 执行验证并记录损失
                net.train()  # 设置网络回到训练模式

                # # 如果当前验证损失小于当前循环的最小ACER，则保存模型
                # if valid_loss[1] < min_acer and epoch > 0:
                #     min_acer = valid_loss[1]  # 更新当前循环的最小ACER
                #     # 构建模型保存路径
                #     ckpt_name = config.save_dir + '/checkpoint/Cycle_' + str(cycle_index) + '_min_acer_model.pth'
                #     torch.save(net.state_dict(), ckpt_name)  # 保存模型权重
                #     log.write('save cycle ' + str(cycle_index) + ' min acer model: ' + str(min_acer) + '\n')  # 记录日志

                # 如果当前验证损失小于全局最小ACER，则保存模型
                # if valid_loss[1] < global_min_acer and epoch > 0:
                #     global_min_acer = valid_loss[1]  # 更新全局最小ACER
                #     # 构建模型保存路径
                #     ckpt_name = config.save_dir + '/checkpoint/global_min_acer_model.pth'
                #     torch.save(net.state_dict(), ckpt_name)  # 保存模型权重
                #     log.write('save global min acer model: ' + str(min_acer) + '\n')  # 记录日志

                if valid_loss[2] > global_max_acc and epoch > 0:
                    global_max_acc = valid_loss[2]  # 更新全局最大ACC
                    # 构建模型保存路径
                    ckpt_name = config.save_dir + '/checkpoint/global_max_acc_model.pth'
                    torch.save(net.state_dict(), ckpt_name)  # 保存模型权重
                    log.write('save global min acc model: ' + str(global_max_acc) + '\n')  # 记录日志

            # 记录每个循环的训练和验证结果
            asterisk = ' '  # 用于标记最好的验证结果
            log.write(model_name + ' Cycle %d: %0.4f %5.1f %6.1f | %0.6f  %0.6f  %0.3f  %0.3f %s  | %0.6f  %0.6f |%s \n' % (
                cycle_index, lr, iter, epoch,
                valid_loss[0], valid_loss[1], valid_loss[2],valid_loss[4], asterisk,
                batch_loss[0], batch_loss[1],
                time_to_str((timer() - start), 'sec')))  # 使用格式化字符串记录日志

        # 循环结束后，保存当前循环的最终模型
        # ckpt_name = config.save_dir + '/checkpoint/Cycle_' + str(cycle_index) + '_final_model.pth'
        # torch.save(net.state_dict(), ckpt_name)  # 保存模型权重
        # log.write('save cycle ' + str(cycle_index) + ' final model \n')  # 记录日志

def run_test(config, dir):
    # 根据配置构建模型名称
    model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)+'_'+config.train_test
    # 设置模型保存的根目录
    config.save_dir = './Models'
    # 根据模型名称创建或指定具体的模型保存目录
    config.save_dir = os.path.join(config.save_dir, model_name)
    # 获取预训练模型的路径
    initial_checkpoint = config.pretrained_model

    ## 初始化模型 -------------------------------
    # 根据配置获取融合模型实例，设置了分类数、图像尺寸和patch尺寸
    net = get_fusion_model(model_name=config.model, num_class=2, image_size=48, patch_size=16)
    # 使用DataParallel来支持多GPU训练
    net = torch.nn.DataParallel(net)
    # 将模型移动到GPU上
    net = net.cuda()



    # 如果指定了预训练模型，则加载预训练模型
    if initial_checkpoint is not None:
        save_dir = os.path.join(config.save_dir + '/checkpoint', dir, initial_checkpoint)
        initial_checkpoint = os.path.join(config.save_dir + '/checkpoint', initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        # 如果目标目录不存在，则创建它
        if not os.path.exists(os.path.join(config.save_dir + '/checkpoint', dir)):
            os.makedirs(os.path.join(config.save_dir + '/checkpoint', dir))

    # # 准备验证集数据
    # valid_dataset = FDDataset_sufr(mode='val', image_size=config.image_size,
    #                                fold_index=config.train_fold_index)
    # valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=config.batch_size,
    #                           drop_last=False, num_workers=8)

    # 准备测试集数据
    test_dataset = FDDataset_wmca(mode='test', image_size=config.image_size,
                                  fold_index=config.train_fold_index)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                             drop_last=False, num_workers=8)

    # 设置损失函数
    # criterion = softmax_cross_entropy_criterion
    # 将模型设置为评估模式
    net.eval()

    # # 在验证集上执行测试，获取损失和输出
    # valid_loss, out = do_valid_test(net, valid_loader, criterion)
    # # 打印验证损失
    # print('%0.6f  %0.6f  %0.3f  (%0.3f) \n' % (valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3]))
    start_time = time.time()
    # 在测试集上执行推理，并获取输出
    out = infer_test(net, test_loader)
    end_time = time.time()
    # 打印完成消息
    print('done')

    # 将测试输出写入到文件，用于提交
    # submission(out, save_dir + '_noTTA.txt', mode='test')

    total_duration = end_time - start_time

    # 打印完成消息和总耗时
    print('Done. Total time taken: {:.2f} seconds.\n'.format(total_duration))

def main(config):
    if config.mode == 'train':
        run_train(config)

    if config.mode == 'infer_test':
        config.pretrained_model = r'global_min_acer_model.pth'
        run_test(config, dir='global_test_36_TTA')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default = -1)
    parser.add_argument('--model', type=str, default='FaceBagNetFusion')
    parser.add_argument('--image_mode', type=str, default='fusion')
    parser.add_argument('--image_size', type=int, default=48)
    parser.add_argument('--patch_size', type=int, default=16)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cycle_num', type=int, default=10)
    parser.add_argument('--cycle_inter', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--train_test',type=str,default='SCP2W',choices=['SCP2W','CWP2S','SCW2P','SCP2W'])

    parser.add_argument('--mode', type=str, default='infer_test', choices=['train','infer_test'])
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./Models')
    parser.add_argument('--dataset_name', type=str, default='4@1', choices=['4@1', '4@2', '4@3'])

    config = parser.parse_args()
    print(config)
    main(config)