import numpy as np
import torch
from scipy import interpolate
from tqdm import tqdm

import numpy as np

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(1 - dist, 1 - threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.shape[0]

    # Calculate FAR and FRR
    far = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    frr = 0 if (tp + fn == 0) else float(fn) / float(tp + fn)

    # Calculate HTER
    hter = (far + frr) / 2

    return tpr, fpr, acc, hter

def calculate(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    return tp,fp,tn,fn

def ACER(threshold, dist, actual_issame):
    tp, fp, tn, fn = calculate(threshold, dist, actual_issame)

    apcer = fp / (tn*1.0 + fp*1.0)
    npcer = fn / (fn * 1.0 + tp * 1.0)
    acer = (apcer + npcer) / 2.0
    return acer,tp, fp, tn,fn

def TPR_FPR( dist, actual_issame, fpr_target = 0.001):
    # acer_min = 1.0
    # thres_min = 0.0
    # re = []

    # Positive
    # Rate(FPR):
    # FPR = FP / (FP + TN)

    # Positive
    # Rate(TPR):
    # TPR = TP / (TP + FN)

    thresholds = np.arange(0.0, 1.0, 0.001)
    nrof_thresholds = len(thresholds)

    fpr = np.zeros(nrof_thresholds)
    FPR = 0.0
    for threshold_idx, threshold in enumerate(thresholds):

        if threshold < 1.0:
            tp, fp, tn, fn = calculate(threshold, dist, actual_issame)
            FPR = fp / (fp*1.0 + tn*1.0)
            TPR = tp / (tp*1.0 + fn*1.0)

        fpr[threshold_idx] = FPR

    if np.max(fpr) >= fpr_target:
        f = interpolate.interp1d(np.asarray(fpr), thresholds, kind= 'slinear')
        threshold = f(fpr_target)
    else:
        threshold = 0.0

    tp, fp, tn, fn = calculate(threshold, dist, actual_issame)

    FPR = fp / (fp * 1.0 + tn * 1.0)
    TPR = tp / (tp * 1.0 + fn * 1.0)

    print(str(FPR)+' '+str(TPR))
    return FPR,TPR

import torch.nn.functional as F
def metric(logit, truth):
    prob = F.softmax(logit, 1)
    value, top = prob.topk(1, dim=1, largest=True, sorted=True)
    correct = top.eq(truth.view(-1, 1).expand_as(top))

    correct = correct.data.cpu().numpy()
    correct = np.mean(correct)
    return correct, prob

def do_valid( net, test_loader, criterion ):
    valid_num  = 0
    losses   = []
    corrects = []
    probs = []
    labels = []

    for input, truth in test_loader:
        b,n,c,w,h = input.size()
        input = input.view(b*n,c,w,h)

        input = input.cuda()
        truth = truth.cuda()

        with torch.no_grad():
            logit = net(input)
            logit = logit.view(b,n,2)
            logit = torch.mean(logit, dim = 1, keepdim = False)

            truth = truth.view(logit.shape[0])
            loss    = criterion(logit, truth, False)
            correct, prob = metric(logit, truth)

        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())
        corrects.append(np.asarray(correct).reshape([1]))
        probs.append(prob.data.cpu().numpy())
        labels.append(truth.data.cpu().numpy())

    # assert(valid_num == len(test_loader.sampler))
    #----------------------------------------------

    correct = np.concatenate(corrects)
    loss    = np.concatenate(losses)
    loss    = loss.mean()
    correct = np.mean(correct)

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    tpr, fpr, acc = calculate_accuracy(0.5, probs[:,1], labels)
    acer,_,_,_,_ = ACER(0.5, probs[:, 1], labels)

    valid_loss = np.array([
        loss, acer, acc, correct
    ])

    return valid_loss,[probs[:, 1], labels]

def do_valid_test_r( net, test_loader, criterion ):
    valid_num  = 0
    losses   = []
    corrects = []
    probs = []
    labels = []

    for i, (input, truth) in enumerate(tqdm(test_loader)):
        b,n,c,w,h = input.size()
        input = input.view(b*n,c,w,h)

        input = input.cuda()
        truth = truth.cuda()

        with torch.no_grad():
            logit = net(input)
            logit = logit.view(b,n,2)
            logit = torch.mean(logit, dim = 1, keepdim = False)

            truth = truth.view(logit.shape[0])
            loss    = criterion(logit, truth, False)
            correct, prob = metric(logit, truth)

        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())
        corrects.append(np.asarray(correct).reshape([1]))
        probs.append(prob.data.cpu().numpy())
        labels.append(truth.data.cpu().numpy())

    correct = np.concatenate(corrects)
    loss    = np.concatenate(losses)
    loss    = loss.mean()
    correct = np.mean(correct)

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

     # 计算准确率、TPR、FPR和ACER
    tpr, fpr, acc,hter = calculate_accuracy(0.5, probs[:, 1], labels)
    acer, _, _, _, _ = ACER(0.5, probs[:, 1], labels)


    # 将计算出的指标整合到一个数组中
    valid_loss = np.array([
        loss, acer, acc, correct,hter
    ])

    # 返回计算出的指标和预测概率与真实标签
    return valid_loss, [probs[:, 1], labels]

def do_valid_test(net, test_loader, criterion):
    # 初始化统计指标
    valid_num = 0  # 有效的样本数量
    losses = []  # 存储每个批次的损失
    corrects = []  # 存储每个批次的正确预测数量
    probs = []  # 存储预测的概率
    labels = []  # 存储真实标签

    # 遍历测试数据加载器
    for i, (input, truth) in enumerate(tqdm(test_loader)):
        b, n, c, w, h = input.size()  # 获取输入数据的尺寸，b:批次大小, n:序列长度, c:通道数, w:宽度, h:高度
        input = input.view(b * n, c, w, h)  # 重新排列输入数据的尺寸，以适应模型

        # 将数据移至GPU上进行加速计算
        input = input.cuda()
        truth = truth.cuda()
        # print("Shape of truth:", truth.shape)

        # 不计算梯度，用于评估和测试
        with torch.no_grad():
            logit = net(input)  # 通过模型获取预测结果
            # logit = logit.view(b, n, 2)  # 将输出结果重新排列，准备进行平均操作
            logit,_ ,_,_,_= net(input)  # 通过模型获取预测结果
            # print("Shape of truth:", truth.shape)
            # print("Shape of logit:", logit.shape)
            # print("Content of input after moving to GPU:")
            # print(logit[:3])  

            # print("Content of truth after moving to GPU:")
            # print(truth)  # 打印truth张量的前三行
            logit = logit.view(b, n, 2)  # 将输出结果重新排列，准备进行平均操作

            # print("Shape of truth:", logit.shape)
            # print("将输出结果重新排列，准备进行平均操作:")
            # print(logit[:3])

            logit = torch.mean(logit, dim=1, keepdim=False)  # 对每个序列的预测结果取平均

            # print("Shape of truth:", logit.shape)
            # print("对每个序列的预测结果取平均:")
            # print(logit[:3])

            truth = truth.view(logit.shape[0])  # 调整真实标签的尺寸以匹配预测结果
            loss = criterion(logit, truth, False)  # 计算损失
            correct, prob = metric(logit, truth)  # 计算正确的预测数量和预测概率

        # 更新统计指标
        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())
        corrects.append(np.asarray(correct).reshape([1]))
        probs.append(prob.data.cpu().numpy())
        labels.append(truth.data.cpu().numpy())

    # 整合各批次的统计指标
    correct = np.concatenate(corrects)
    loss = np.concatenate(losses)
    loss = loss.mean()  # 计算平均损失
    correct = np.mean(correct)  # 计算平均正确率

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    # 计算准确率、TPR、FPR和ACER
    tpr, fpr, acc,hter = calculate_accuracy(0.5, probs[:, 1], labels)
    acer, _, _, _, _ = ACER(0.5, probs[:, 1], labels)


    # 将计算出的指标整合到一个数组中
    valid_loss = np.array([
        loss, acer, acc, correct,hter
    ])

    # 返回计算出的指标和预测概率与真实标签
    return valid_loss, [probs[:, 1], labels]
def infer_test( net, test_loader):
    valid_num  = 0
    probs = []

    for i, (input, truth) in enumerate(tqdm(test_loader)):
        b,n,c,w,h = input.size()
        input = input.view(b*n,c,w,h)
        input = input.cuda()

        with torch.no_grad():
            # logit,_,_   = net(input)
            # print(net(input))
            # break
            logit,_,_,_,_ = net(input)
            
            logit = logit.view(b,n,2)
            logit = torch.mean(logit, dim = 1, keepdim = False)
            prob = F.softmax(logit, 1)

        valid_num += len(input)
        probs.append(prob.data.cpu().numpy())

    probs = np.concatenate(probs)
    return probs[:, 1]



