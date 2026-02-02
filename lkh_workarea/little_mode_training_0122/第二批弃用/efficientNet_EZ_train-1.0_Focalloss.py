"""
Evaluate on ImageNet. Note that at the moment, training is not implemented (I am working on it).
that being said, evaluation is working.
"""

import argparse
import os
import random
import shutil
import time
import warnings
import PIL
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import Lambda, InterpolationMode
from torchvision.transforms.v2 import RandAugment
from efficientnet_pytorch import EfficientNet

# 导入高级损失函数
from focal_loss_training import FocalLoss, AdaptiveWeightedLoss, OnlineHardExampleMining, CombinedLoss

class_counts = [54561, 27280]
weights = torch.tensor(class_counts)
weights = (1.0 / weights) / (1.0 / weights).sum()  # 归一化


def hsv_transform(image):
    # 生成随机增益
    r = np.random.uniform(-1, 1, 3) * [0.015, 0.7, 0.4] + 1  # H, S, V增益范围
    hue, sat, val = r

    # 调整色调（H）
    h = torch.Tensor([hue])  # 转换为张量操作
    # 调整饱和度（S）和亮度（V）
    image = image * torch.Tensor([val])
    image = torch.clamp(image, 0, 1)

    return image


transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为Tensor
    # 其他预处理（如归一化、LetterBox等）
])


class Cutout:
    def __init__(self, n_holes=1, length=80):
        """
        Args:
            n_holes (int): 遮挡区域数量
            length (int): 遮挡区域边长（正方形）
        """
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): 输入图像张量，形状为 [C, H, W]
        Returns:
            Tensor: 应用遮挡后的图像张量
        """
        h = img.size(1)  # 图像高度
        w = img.size(2)  # 图像宽度

        # 生成与图像尺寸相同的全1掩码
        mask = torch.ones((h, w), dtype=torch.float32, device=img.device)

        for _ in range(self.n_holes):
            # 随机生成遮挡中心坐标
            y = torch.randint(low=0, high=h, size=(1,)).item()
            x = torch.randint(low=0, high=w, size=(1,)).item()
            #
            # # 计算遮挡区域边界
            # y1 = torch.clamp(y - self.length // 2, min=0, max=h)
            # y2 = torch.clamp(y + self.length // 2, min=0, max=h)
            # x1 = torch.clamp(x - self.length // 2, min=0, max=w)
            # x2 = torch.clamp(x + self.length // 2, min=0, max=w)
            # 转换为张量
            y_tensor = torch.tensor(y, dtype=torch.int32, device=img.device)
            x_tensor = torch.tensor(x, dtype=torch.int32, device=img.device)

            # 正确调用 clamp
            y1 = torch.clamp(y_tensor - self.length // 2, min=0, max=y_tensor)
            y2 = torch.clamp(y_tensor + self.length // 2, min=0, max=y_tensor)
            x1 = torch.clamp(x_tensor - self.length // 2, min=0, max=x_tensor)
            x2 = torch.clamp(x_tensor + self.length // 2, min=0, max=x_tensor)
            # 将遮挡区域置0
            mask[y1:y2, x1:x2] = 0.

        # 扩展掩码维度以匹配图像通道数 [C, H, W]
        mask = mask.unsqueeze(0)

        # 应用遮挡（保留原始数据类型）
        img = img * mask

        return img

class YOLO_HSV_Augment:
    def __init__(self, h_gain=0.015, s_gain=0.7, v_gain=0.4):
        """
        YOLO风格RGB空间颜色增强
        参数:
            h_gain: 色调调整幅度:ml-citation{ref="7" data="citationList"}
            s_gain: 饱和度调整幅度:ml-citation{ref="7" data="citationList"}
            v_gain: 亮度调整幅度:ml-citation{ref="7" data="citationList"}
        """
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain

    def __call__(self, img_tensor):
        """
        输入:
            img_tensor: 标准化后的图像张量 (C,H,W), 值域[0,1]
        返回:
            augmented: 增强后的图像张量
        """
        # 生成随机增益系数
        r = np.random.uniform(-1, 1, 3) * [self.h_gain, self.s_gain, self.v_gain] + 1
        h, s, v = r

        # 亮度调整（所有通道乘增益）
        img_tensor *= v
        img_tensor = torch.clamp(img_tensor, torch.tensor(0), torch.tensor(1))

        # 饱和度调整（基于相对亮度）
        luma = 0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]
        for c in range(3):
            img_tensor[c] = luma + (img_tensor[c] - luma) * s
        img_tensor = torch.clamp(img_tensor, 0, 1)

        # 色调调整（通道混合）
        mix = np.random.uniform(-h, h, 3)
        mix_matrix = torch.tensor([
            [1 + mix[0], -mix[0], 0],
            [-mix[1], 1 + mix[1], 0],
            [0, -mix[2], 1 + mix[2]]
        ], dtype=torch.float32)
        img_tensor = torch.einsum('ij,jkl->ikl', mix_matrix, img_tensor)
        return torch.clamp(img_tensor, 0, 1)


# -------------------- 早停机制 --------------------
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset', default='/home/si2/HYM_DATA/boiling_over_imagenet_format')
parser.add_argument('-data',
                    help='path to dataset', default=r'砂锅分类数据集(7-18)服务器')
parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b0',
                    help='model architecture (default: resnet18)')
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
#                     help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('--resume', default='/home/si2/duyuelai/EfficientNet-PyTorch-master/examples/', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='use advprop or not')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0.0


def main():
    args = parser.parse_args()
    args.data = r"/root/autodl-tmp/pot_dataset/pot_dataset"
    args.pretrained = True
    args.batch_size = 64 #256
    args.epochs = 200
    args.gpu = 0  #一张卡
    # args.evaluate = True
    # args. = '/home/si2/HYM_DATA/boiling_over_imagenet_format'
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a spefic GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if 'efficientnet' in args.arch:  # NEW
        if args.pretrained:
            model = EfficientNet.from_pretrained(args.arch, advprop=args.advprop, num_classes=2)
            print("=> using pre-trained model '{}'".format(args.arch))
        else:
            print("=> creating model '{}'".format(args.arch))
            # model = EfficientNet.from_name(args.arch)
            model = EfficientNet.from_name(args.arch)
            in_features = model._fc.in_features
            model._fc = nn.Linear(in_features, 2)

    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

    # in_feature = model._fc.in_features
    # model._fc = nn.Linear(in_feature, 2)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            if torch.cuda.is_available():
                model.cuda()
            else:
                model.cpu()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.cuda.is_available():
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        # 在CPU上运行
        model = model.cpu()

    # define loss function (criterion) and optimizer
    # 使用Focal Loss替代原来的CrossEntropyLoss
    criterion = FocalLoss(
        alpha=0.75,      # 溢出类权重
        gamma=2.0,       # 聚焦难样本
        reduction='mean'
    )
    
    # 根据设备类型分配损失函数
    if args.gpu is not None and torch.cuda.is_available():
        criterion = criterion.cuda(args.gpu)
    elif torch.cuda.is_available():
        criterion = criterion.cuda()
    else:
        criterion = criterion.cpu()
    
    # 可选：使用其他高级损失函数
    # 1. 自适应权重损失
    # criterion = AdaptiveWeightedLoss(class_counts=class_counts, temperature=1.0)
    # criterion = criterion.cuda(args.gpu) if torch.cuda.is_available() else criterion.cpu()
    
    # 2. 在线难样本挖掘
    # criterion = OnlineHardExampleMining(keep_ratio=0.7)
    # criterion = criterion.cuda(args.gpu) if torch.cuda.is_available() else criterion.cpu()
    
    # 3. 组合损失
    # criterion = CombinedLoss(focal_weight=0.7, ce_weight=0.3)
    # criterion = criterion.cuda(args.gpu) if torch.cuda.is_available() else criterion.cpu()

    # 使用优化的AdamW配置
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,         # 比模型2更保守的学习率
        weight_decay=5e-4  # 更强的正则化
    )
    
    # 配置学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,      # 学习率衰减因子
        patience=6,      # 等待轮数
        min_lr=1e-6,     # 最小学习率
        # 在新版本PyTorch中，verbose参数已移除
    )

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    
    # 检查数据路径是否存在
    if not os.path.exists(traindir) or not os.path.exists(valdir):
        print(f"错误：数据路径不存在！")
        print(f"训练路径: {traindir}")
        print(f"验证路径: {valdir}")
        print(f"请修改args.data参数，指向正确的数据目录。")
        return
    if args.advprop:
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                     std=[1, 1, 1])

    if 'efficientnet' in args.arch:
        image_size = EfficientNet.get_image_size(args.arch)
    else:
        image_size = args.image_size
    #image_size = 224
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7,1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandAugment(num_ops=2, magnitude=9, num_magnitude_bins=31, interpolation=InterpolationMode.BILINEAR, fill=None),
            # transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize,
            # transforms.Normalize([0.,0.,0.], [1.,1.,1.]),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2,scale=(0.02,0.15),ratio=(0.3,3.3)),
            # YOLO_HSV_Augment(),  # 自定义增强

            # Cutout(),
            # normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_transforms = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    print('Using image size', image_size)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        res = validate(val_loader, model, criterion, args)
        with open('res.txt', 'w') as f:
            print(res, file=f)
        return

    early_stopper = EarlyStopper(patience=15, min_delta=0.003)
    use_cos = False

    if use_cos:
        scheduler = None
        scheduler = adjust_learning_rate(optimizer,  args.epochs, args, 'cos')

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if use_cos:
            adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, history)

        if use_cos:
            if scheduler is not None:
                scheduler.step()

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, history, scheduler)
        if early_stopper(history['val_loss'][-1]):
            print(f'early stopping triggered at epoch {epoch}')
            break
        a = optimizer.param_groups[0]['lr']
        print('lr=', a)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            print('epoch, best_acc1, acc1, is_best', epoch, best_acc1, acc1, is_best)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            })
            if is_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, filename=f'model_best{best_acc1:.4f}.pth.tar')
    visualize_results(history)

def train(train_loader, model, criterion, optimizer, epoch, args, history=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    
    # switch to train mode
    model.train()
    running_loss = 0.0
    running_corrects = 0

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        elif torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        else:
            images = images.cpu()
            target = target.cpu()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 1))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

        # 统计运行损失和正确率
        _, preds = torch.max(output, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == target.data)

    # 计算 epoch 级别的损失和准确率
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    # 更新历史记录
    if history is not None:
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
    
    # 打印 epoch 总结
    print(f"Epoch {epoch} Summary: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc.item():.4f}")
    
    return epoch_loss, epoch_acc

def validate(val_loader, model, criterion, args, history, scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None and torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            elif torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            else:
                images = images.cpu()
                target = target.cpu()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

            _, preds = torch.max(output, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == target.data)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)

        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc.item())

        # 学习率调度
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

    return top1.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    保存模型的训练快照，同时如果是最佳模型，则额外保存权重用于推理。

    参数:
        state: 字典，包含 epoch、state_dict、optimizer 等信息
        filename: 保存文件名（默认 checkpoint.pth.tar）
    """
    # 保存训练快照（完整状态）
    ckpt_dir = r"/root/autodl-tmp/model_V12.2b"
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, filename)
    torch.save(state, ckpt_path)
    print(f"✅ Checkpoint saved to: {ckpt_path}")

    # 如果保存的是 best 模型，额外导出纯权重文件（.pth）
    if filename.startswith('model_best'):
        weight_path = os.path.join(ckpt_dir, 'efficientnet_b0_best.pth')
        torch.save(state['state_dict'], weight_path)
        print(f"🎯 Best model weights saved to: {weight_path}")


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


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args, flag='steplr'):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if flag =='steplr':
        lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif flag=='cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-4)
        return scheduler


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# -------------------- 可视化 --------------------
def visualize_results(history):
    import matplotlib.pyplot as plt

    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']

    epochs = range(1, len(train_loss) + 1)

    # ----------- 1. 绘制 Loss -----------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Val Loss', marker='o')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # ----------- 2. 绘制 Accuracy -----------
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Acc', marker='o')
    plt.plot(epochs, val_acc, label='Val Acc', marker='o')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    # ----------- 3. 自动判断是否过拟合/欠拟合 -----------
    if len(train_loss) >= 5:
        final_train_loss = train_loss[-1]
        final_val_loss = val_loss[-1]
        final_train_acc = train_acc[-1]
        final_val_acc = val_acc[-1]

        print("\n📈 Final Performance:")
        print(f"Train Loss: {final_train_loss:.4f}, Val Loss: {final_val_loss:.4f}")
        print(f"Train Acc : {final_train_acc:.4f}, Val Acc : {final_val_acc:.4f}")

        # 分析
        print("\n🧠 Diagnostic Suggestion:")
        if final_train_loss < final_val_loss * 0.7 and final_train_acc > final_val_acc + 0.15:
            print("⚠️ 可能过拟合：训练表现明显优于验证，请尝试正则化或数据增强")
        elif final_train_loss > final_val_loss and final_train_acc < final_val_acc:
            print("❓ 可能欠拟合：模型未充分学习，可以增加训练轮数或使用更复杂模型")
        else:
            print("✅ 模型训练状态正常")

    else:
        print("训练 epoch 太少，无法评估是否过拟合/欠拟合")


if __name__ == '__main__':
    main()
