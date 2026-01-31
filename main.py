import torch.optim as optim
from model import DF,get_matrix
import numpy as np
import time
import configparser
import torch
from tqdm import tqdm
import json
from get_base_feature import get_base
import time
from torchsummary import summary
import numpy as np
import torch as t
from model import DF
import shutil
import random
from util.data_generator import *
import torch.nn as nn
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from dotmap import DotMap
from util.circle_loss import *
from sklearn.model_selection import ParameterGrid

# 设置随机种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

#多视角
def temporal_mask_batch_tail(x, min_ratio=0.1, max_ratio=0.2):
    """
    x: [B, C, T]
    掩码连续的最后部分（尾部有效数据），mask 长度 ∈ [min_ratio, max_ratio] * 实际数据长度
    补齐的 0 不受影响
    """
    B, C, T = x.shape
    x = x.clone()

    for i in range(B):
        # 找非零数据的最后位置
        nonzero_idx = (x[i,0,:] != 0).nonzero(as_tuple=True)[0]
        if len(nonzero_idx) == 0:
            continue

        actual_len = len(nonzero_idx)
        mask_ratio = np.random.uniform(min_ratio, max_ratio)
        mask_len = max(int(actual_len * mask_ratio), 1)

        # 掩码尾部连续 mask
        start = nonzero_idx[-mask_len]  # 从最后 mask_len 个开始
        x[i, :, start:] = 0

    return x




def read_conf(file):
    cf = configparser.ConfigParser()
    cf.read(file)  
    return dict(cf['default'])


# 读取 .npz 中存储的特征和标签
def loadData(fpath):
    train = np.load(fpath,allow_pickle=True).item()
    train_X ,train_y = train['feature'], train['label']
    return train_X, train_y

# 实现了一个简单的 EarlyStopping，防止训练过拟合，用于 train_base()
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
            not improving
        :param min_delta: minimum difference between new loss and old loss for
            new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def top1accuracy(pred, target):
    """Computes the precision@1"""
    batch_size = target.size(0)

    correct = pred.eq(target).float().sum(0)
    return correct.mul_(100.0 / batch_size)

def is_valid_mixture(mixture):
    """Check if mixture is a 2D array with strings representing the models."""
    assert type(mixture) == list and len(mixture) > 0
    for inner_comb in mixture:
        assert type(inner_comb) == list and len(inner_comb) > 0
        for model in inner_comb:
            assert model in ['dir', 'time', 'metadata']

# 训练阶段 3：train_novel()，对于新类别（novel class）做少样本分类。
def train_novel( model,num_classes,source_domain,dataset,shot,include=None):
    """Train and validate model.
    加载 meta 训练后的模型 DF
    只训练一个线性分类器：clf = nn.Linear(2560, num_classes)
    训练 support set（shot 样本）
    评估 query set
    """
    args = DotMap()
    args.gpu = 0 
    epochs = 500
    args.num_workers = 16
    args.query = 70
    args.shot = shot
    train_time_start = time.time()
    clf = nn.Sequential(nn.Linear(2560,num_classes)).cuda()
    model.load_state_dict(torch.load(f'./pre_trained_model/finish_model_{dataset}_{shot}.pt'))
    clf.load_state_dict(torch.load(f'./pre_trained_model/clf_model_{dataset}_{shot}.pt'))

    gpu = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(clf.parameters(),lr=0.01)

    if gpu is not None: 
        t.cuda.set_device(gpu)
        model = model.cuda(gpu)
    model.eval()
    support_loader,query_loader = get_new_dataloader(args, dataset, num_classes) # 修改
    for epoch in tqdm(range(epochs),desc='Epoch'):
        clf.train(True)
        losses = AverageMeter('Loss')
        for i, batch in enumerate(support_loader):
            data, train_label = [_.cuda(gpu, non_blocking=True) for _ in batch]

            data = data.float()
            # data = data.unsqueeze(1)   # [B, 1, 5000]不用，因为datasets里已经unsqueeze过一次了

            # with torch.no_grad():
            #     feat = model(data)     # [B, 2560]
            # 我的修改
            M = 5  # 多视角数量

            with torch.no_grad():
                feats = []
                for _ in range(M):
                    data_aug = temporal_mask_batch_tail(data, min_ratio=0.1, max_ratio=0.2)
                    feat_aug = model(data_aug)          # [B, 2560]
                    feats.append(feat_aug)

                feats = torch.stack(feats, dim=0)       # [M, B, 2560]
                feat = feats.mean(dim=0)                # [B, 2560]

            optimizer.zero_grad()
            output = clf(feat)
            loss = criterion(output, train_label)

            loss.backward()
            optimizer.step()

            losses.update(loss.item(), data.size(0))
        if (epoch % 50 == 0):
            print('training_loss/pretrain_CEL',losses.avg,epoch)
            # torch.save(model.state_dict(), './pre_trained_model/fine_model_DF19.pt')
            loo, acc ,tpr,fpr,f1= validation_novel(model,clf,criterion,query_loader)   
            print('test: {}, loss: {}, TPR: {}, FPR: {}, F1: {}'.format(acc,loo,tpr,fpr,f1))

    print('training_loss/pretrain_CEL',losses.avg,epoch)
    # torch.save(model.state_dict(), './pre_trained_model/fine_model_DF19.pt')
    loo, acc ,tpr,fpr,f1= validation_novel(model,clf,criterion,query_loader)   
    print('test: {}, loss: {}, TPR: {}, FPR: {}, F1: {}'.format(acc,loo,tpr,fpr,f1))
    # torch.save(clf.state_dict(),f'./pre_trained_model/open_model_{dataset}_{shot}.pt')

def get_gradient(module, grad_input, grad_output):
    # 指定模块的梯度
    # print('梯度:', grad_output)
    pass

# 训练阶段 2：train_meta()，对 DF 的 SE attention 模块 做元优化
def train_meta( model,num_classes,dataset,shot,include=None):
    """Train and validate model.
    只训练 attention 模块，不动 backbone
    学习如何在不同 domain 少样本情况下提取有用特征"""
     
    args = DotMap()
    args.gpu = 0 
    epochs = 500
    args.num_workers = 16
    args.query = 70
    args.shot = shot
    train_time_start = time.time()
    clf = nn.Sequential(nn.Linear(2560,num_classes)).cuda()
    gpu = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(clf.parameters(),lr=0.01)
    optimizer_attention = torch.optim.SGD(model.se_attention.parameters(),lr=0.01)
    if gpu is not None: 
        t.cuda.set_device(gpu)
        model = model.cuda(gpu)
    support_loader,query_loader = get_meta_dataloader(args,dataset,num_classes) # 
    hook = model.se_attention.register_backward_hook(get_gradient)
    model.eval()
    for epoch in tqdm(range(epochs),desc='Epoch'):
        clf.train(True)
        losses = AverageMeter('Loss')
        for i, batch in enumerate(support_loader):
            data, train_label = [_.cuda(gpu,non_blocking=True) for _ in batch]

        # forward pass
            optimizer_attention.zero_grad()
            optimizer.zero_grad()
            output = model(data)
            output = clf(output)
            
            loss = criterion(output,train_label)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer_attention.step()
        
            losses.update(loss.item(),data.size(0))
        if (epoch % 50 == 0):
            print('training_loss/pretrain_CEL',losses.avg,epoch)
            # torch.save(model.state_dict(), './pre_trained_model/fine_model_DF19.pt')
            loo, acc ,tpr,fpr,f1 = validation(model,clf,criterion,query_loader)   
            print('test: {}, loss: {}, TPR: {}, FPR: {}, F1: {}'.format(acc,loo,tpr,fpr,f1))
    loo, acc ,tpr,fpr,f1= validation(model,clf,criterion,query_loader)   
    print('test: {}, loss: {}, TPR: {}, FPR: {}, F1: {}'.format(acc,loo,tpr,fpr,f1))
    torch.save(model.state_dict(), f'./pre_trained_model/finish_model_{dataset}_{shot}.pt')
    torch.save(clf.state_dict(), f'./pre_trained_model/clf_model_{dataset}_{shot}.pt')
    train_time_end = time.time()
    print('Total training time: %f' % (train_time_end - train_time_start))

# 训练阶段 1：基于 CircleLoss 的 metric learning，使用的 DF模型，也就是文章里的预训练
def train_base(model,m,gamma,source_domain):
    """Train and validate model.
     加载训练数据：get_base_dataloader(source_domain)
     单独使用 CircleLoss 训练 DF 网络（不训练分类器）
     EarlyStopping 控制停止
     训练结果：输出一个预训练特征提取器：./pre_trained_model/finish_model_{m}_{gamma}_{source_domain}.pt"""
     
    early = EarlyStopping(512)
    train_time_start = time.time()

    criterion = CircleLoss(m,gamma)
    gpu = 0
    epochs = 8000

    if gpu is not None: 
        t.cuda.set_device(gpu)
        model = model.cuda(gpu)
        criterion = criterion.cuda(gpu)

    optimizer = optim.SGD(model.parameters(),lr=0.1)

    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.5)
    train_loader = get_base_dataloader(source_domain)
    train_iterator = iter(train_loader)
    for epoch in tqdm(range(epochs),desc='Epoch'):
        global_count = 0

        losses = AverageMeter('Loss')
        acc = AverageMeter('Acc@1')

        model.train(True)
        batch = next(train_iterator)

        global_count = global_count + 1
        data, train_label = [_.cuda(gpu,non_blocking=True) for _ in batch]
        # forward pass
        optimizer.zero_grad()
        output = model(data)
        inp_sp,inp_sn = convert_label_to_similarity(output,train_label)
        loss = criterion(inp_sp,inp_sn)
        # Backpropagation
        loss.backward()
        early(loss.item())
        optimizer.step()
        accuracy = top1accuracy(output.argmax(dim=1),train_label)
        losses.update(loss.item(),data.size(0))
        acc.update(accuracy.item(), data.size(0))
        if(early.early_stop):
            break
        # lr_scheduler.step()  

        if(epoch % 500 == 0):
            print('training_loss/pretrain_CEL',losses.avg,epoch)

    # torch.save(model.state_dict(), f'./pre_trained_model/finish_model_{m}_{gamma}_{source_domain}.pt')

    train_time_end = time.time()

    print('Total training time: %f' % (train_time_end - train_time_start))

""" 计算 accuracy
计算 TPR/FPR/F1（通过 get_matrix()）
输出 mean TPR, FPR, F1
"""
def validation_novel(model, cls, criterion, dataloader):
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')

    pre = []
    target = []

    model.eval()
    cls.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            data, label = [_.cuda(0, non_blocking=True) for _ in batch]

            # ===== 核心修正点 =====
            data = data.float()
            if data.dim() == 2:
                data = data.unsqueeze(1)      # [B, 1, 5000]

            feat = model(data)               # [B, 2560]
            output = cls(feat)               # [B, num_classes]
            # =====================

            pre.extend(output.argmax(dim=1).cpu())
            target.extend(label.cpu())

            loss = criterion(output, label)
            accuracy = top1accuracy(output.argmax(dim=1), label)

            losses.update(loss.item(), data.size(0))
            acc.update(accuracy.item(), data.size(0))

    pre = torch.tensor(pre, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)

    TPR, FPR, F1 = get_matrix(pre, target)

    return losses.avg, acc.avg, TPR.mean(), FPR.mean(), F1.mean()


def validation(model,cls,criterion,dataloader):
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')
    pre = []
    target = []
    model.eval()
    with t.no_grad(): 
        for i, batch in enumerate(dataloader):
            data, label = [_.cuda(0,non_blocking=True) for _ in batch]

            output = model(data)
            output = cls(output)
            pre.extend(output.argmax(dim=1).cpu())
            target.extend(label.cpu())
            loss = criterion(output,label)
            accuracy = top1accuracy(output.argmax(dim=1),label)
            losses.update(loss.item(),data.size(0))
            acc.update(accuracy.item(),data.size(0))
    pre = torch.tensor(pre,dtype=torch.float32)
    
    target = torch.tensor(target,dtype=torch.float32)
    TPR,FPR,F1 = get_matrix(pre,target)
    # acc_pre = pre.eq(target).float()
    # acm = {}
    # for label in np.unique(target):
    #     inds = np.argwhere(target == label)
    #     acm[label] = acc_pre[inds].sum().sum()/95
    # Plot figure if needed
    # print(acm)
    return losses.avg, acc.avg,TPR.mean(),FPR.mean(),F1.mean()


setup_seed(42)
model = DF()
model.cuda()
# model.hook_register()

summary(model,(1,5000))

m = 0.5
gamma = 64
# dataset = 'tor_100w_2500tr'
source_domain = 'AWF_775'
# train_base(model,m,gamma,source_domain)
# num_class = 100

# model.load_state_dict(torch.load(f'./pre_trained_model/finish_model_{m}_{gamma}_{source_domain}.pt'))
# train_meta(model,num_classes=num_class,dataset=dataset,shot=5)
# get_base(source_domain,dataset,1)
# train_novel(model,num_classes=num_class,source_domain=source_domain,dataset=dataset,shot=1)
# dataset = 'DF19_WTFPAD'
# source_domain = 'KNN_WTFPAD'
# for shot in [1,5,10,15,20]:
#     print('shot',shot)
#     shot = shot
#     model.load_state_dict(torch.load(f'./pre_trained_model/finish_model_{m}_{gamma}_{source_domain}.pt'))
#     train_meta(model,num_classes=num_class,dataset=dataset,shot=shot)
#     get_base(source_domain,dataset,shot)
#     train_novel(model,num_classes=num_class,source_domain=source_domain,dataset=dataset,shot=shot)

# for include in [50,75,100]:
#     print('include',include)
#     for shot in [1,5,10,15,20]:
#         print('shot',shot)
#         shot = shot
#         model.load_state_dict(torch.load(f'./pre_trained_model/finish_model_{m}_{gamma}.pt'))
#         train_meta(model,num_classes=num_class,dataset=dataset,shot=shot,include=include)
#         get_base(dataset,shot)
#         train_novel(model,num_classes=num_class,dataset=dataset,shot=shot,include=include)

# for dataset,num_class in zip(['tor_time_test3d_200w_100tr','tor_time_test10d_200w_100tr','tor_time_test2w_200w_100tr','tor_time_test4w_200w_100tr','tor_time_test6w_200w_100tr'],\
#                              [200,200,200,200,199]):
# for dataset,num_class in zip(['tor_100w_2500tr','KNN','DF19','DF95'],\
#                              [100,100,100,95]):
#     print('dataset',dataset)
#     for shot in [1,5,10,15,20]:
#         print('shot',shot)
#         shot = shot
#         model.load_state_dict(torch.load(f'./pre_trained_model/finish_model_{m}_{gamma}.pt'))
#         train_meta(model,num_classes=num_class,dataset=dataset,shot=shot)
#         get_base(source_domain,dataset,shot)
#         train_novel(model,num_classes=num_class,source_domain=source_domain,dataset=dataset,shot=shot)


#ablation [meta,calibration]
""" 最后的代码是为了跑消融实验：
1.加载基础模型
2.元训练（train_meta）
3.少样本训练（train_novel）
"""
for dataset,num_class in zip(['tor_100w_2500tr'],\
                             [100]):
    print('dataset',dataset)
    for shot in [1,5,10,15,20]:
        print('shot',shot)
        shot = shot
        model.load_state_dict(torch.load(f'./pre_trained_model/finish_model_{m}_{gamma}.pt'))
        train_meta(model,num_classes=num_class,dataset=dataset,shot=shot)
        get_base(source_domain,dataset,shot)
        train_novel(model,num_classes=num_class,source_domain=source_domain,dataset=dataset,shot=shot)

