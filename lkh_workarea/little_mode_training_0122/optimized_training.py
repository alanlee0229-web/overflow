"""
方案4: 集成所有优化的完整训练代码
基于你的模型2，添加：
1. Focal Loss
2. 更激进的正类权重
3. 数据增强微调
4. 混合精度训练
5. 梯度累积

预期: Recall 97-98%, F1 97.5+%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from efficientnet_pytorch import EfficientNet

# ============ 核心Loss函数 ============
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        alpha_weight = torch.where(
            targets == 1,
            torch.tensor(self.alpha).to(inputs.device),
            torch.tensor(1 - self.alpha).to(inputs.device)
        )
        
        loss = alpha_weight * focal_weight * ce_loss
        return loss.mean()


# ============ 优化的训练函数 ============
def train_epoch_optimized(train_loader, model, criterion, optimizer, 
                         epoch, args, scaler, accumulation_steps=2):
    """
    优化的训练epoch
    
    Args:
        accumulation_steps: 梯度累积步数（模拟更大batch）
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    optimizer.zero_grad()
    
    from tqdm import tqdm
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for i, (images, targets) in enumerate(pbar):
        images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)
        
        # 混合精度训练
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps  # 梯度累积需要平均
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # 统计
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0) * accumulation_steps
        running_corrects += torch.sum(preds == targets.data)
        total_samples += images.size(0)
        
        pbar.set_postfix({
            'loss': f'{running_loss / total_samples:.4f}',
            'acc': f'{running_corrects.double() / total_samples:.4f}'
        })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()


# ============ 优化的验证函数 ============
def validate_optimized(val_loader, model, criterion, args, return_details=False):
    """优化的验证函数，返回详细指标"""
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # 计算详细指标
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score, precision_score, recall_score)
    
    epoch_loss = running_loss / len(val_loader.dataset)
    
    metrics = {
        'loss': epoch_loss,
        'f1': f1_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'accuracy': sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    }
    
    if return_details:
        print("\n" + classification_report(all_labels, all_preds,
                                          target_names=['Normal', 'Overflow'],
                                          digits=4))
        cm = confusion_matrix(all_labels, all_preds)
        print(f"\nConfusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    
    return metrics, all_probs, all_labels


# ============ 主训练函数（替换你的main_worker） ============
def main_worker_optimized(gpu, ngpus_per_node, args):
    """完全优化的训练流程"""
    global best_acc1
    args.gpu = gpu
    
    # ========== 模型初始化 ==========
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")
    
    model = EfficientNet.from_pretrained(
        args.arch,
        advprop=args.advprop,
        num_classes=2
    )
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    
    # ========== Loss函数选择 ==========
    # 方案A: Focal Loss（推荐）
    criterion = FocalLoss(alpha=0.80, gamma=2.0).cuda(args.gpu)
    
    # 方案B: 加权交叉熵（备选）
    # class_counts = [54561, 27280]
    # weights = torch.tensor([0.25, 0.75]).cuda()  # 更激进的正类权重
    # criterion = nn.CrossEntropyLoss(weight=weights).cuda(args.gpu)
    
    # ========== 优化器配置 ==========
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=5e-4,
        betas=(0.9, 0.999)
    )
    
    # ========== 学习率调度器 ==========
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=6,
        min_lr=1e-6,
        verbose=True
    )
    
    # ========== 混合精度训练 ==========
    scaler = GradScaler()
    
    # ========== 数据加载 ==========
    import os
    from torchvision import transforms, datasets
    import PIL
    
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    
    # 优化的数据增强
    normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1, 1, 1])
    image_size = EfficientNet.get_image_size(args.arch)
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.7, 1.0),
            ratio=(0.85, 1.15),
            antialias=True
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15  # 稍微提高
        ),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(
            p=0.25,  # 稍微提高
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3)
        )
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    train_dataset = datasets.ImageFolder(traindir, train_transforms)
    val_dataset = datasets.ImageFolder(valdir, val_transforms)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # ========== 训练循环 ==========
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_f1': [], 'val_recall': [], 'val_precision': []
    }
    
    best_f1 = 0.0
    patience_counter = 0
    max_patience = 15
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print('='*60)
        
        # 训练
        train_loss, train_acc = train_epoch_optimized(
            train_loader, model, criterion, optimizer,
            epoch, args, scaler, accumulation_steps=2
        )
        
        # 验证
        val_metrics, _, _ = validate_optimized(
            val_loader, model, criterion, args,
            return_details=(epoch % 5 == 0)  # 每5个epoch打印详细信息
        )
        
        # 学习率调度
        scheduler.step(val_metrics['loss'])
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_precision'].append(val_metrics['precision'])
        
        # 打印指标
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}, "
              f"Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存最佳模型
        is_best = val_metrics['f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_f1': best_f1,
                'optimizer': optimizer.state_dict(),
                'metrics': val_metrics
            }, filename=f'model_best_f1_{best_f1:.4f}.pth.tar')
            
            print(f"✅ New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early Stopping
        if patience_counter >= max_patience:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break
    
    # 可视化
    visualize_training_history(history)
    
    return model, history


# ============ 可视化函数 ============
def visualize_training_history(history):
    """可视化训练历史"""
    import matplotlib.pyplot as plt
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy曲线
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[0, 1].set_title('Accuracy Curve')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1曲线
    axes[1, 0].plot(epochs, history['val_f1'], 'g-', label='Val F1')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Precision & Recall
    axes[1, 1].plot(epochs, history['val_precision'], 'b-', label='Precision')
    axes[1, 1].plot(epochs, history['val_recall'], 'r-', label='Recall')
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_optimized.png', dpi=300)
    print("\n📊 训练曲线已保存至 training_history_optimized.png")


# ============ 检查点保存 ============
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    import os
    ckpt_dir = "/root/autodl-tmp/model_optimized"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    ckpt_path = os.path.join(ckpt_dir, filename)
    torch.save(state, ckpt_path)
    print(f"✅ Checkpoint saved: {ckpt_path}")
    
    # 额外保存纯权重
    if 'best' in filename:
        weight_path = os.path.join(ckpt_dir, 'best_model_weights.pth')
        torch.save(state['state_dict'], weight_path)
        print(f"🎯 Weights saved: {weight_path}")
