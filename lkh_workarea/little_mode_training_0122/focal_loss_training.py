"""
方案2: Focal Loss + 自适应类别权重
优点: 专门针对难样本和类别不平衡，显著降低漏检
需要: 重新训练
预期提升: Recall +2-4%, 总体F1 +1-2%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss - 专治难分样本和类别不平衡"""
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 正类权重（0.75表示更关注溢出类）
            gamma: 聚焦参数（2.0是标准值，越大越关注难样本）
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) 模型输出logits
            targets: (N,) 真实标签
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 预测概率
        
        # Focal权重
        focal_weight = (1 - pt) ** self.gamma
        
        # 类别权重
        alpha_weight = torch.where(
            targets == 1,
            torch.tensor(self.alpha).to(inputs.device),
            torch.tensor(1 - self.alpha).to(inputs.device)
        )
        
        loss = alpha_weight * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AdaptiveWeightedLoss(nn.Module):
    """自适应类别权重损失"""
    def __init__(self, class_counts, temperature=1.0):
        """
        Args:
            class_counts: [正常样本数, 溢出样本数]
            temperature: 温度参数，控制权重平滑度
        """
        super(AdaptiveWeightedLoss, self).__init__()
        
        # 计算自适应权重
        total = sum(class_counts)
        weights = torch.tensor([
            total / (2 * class_counts[0]),
            total / (2 * class_counts[1])
        ]) ** temperature
        
        self.register_buffer('weights', weights)
    
    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, weight=self.weights)


# ============ 完整训练配置（方案2A：Focal Loss） ============
def get_focal_loss_config():
    """Focal Loss训练配置"""
    return {
        'loss_fn': FocalLoss(alpha=0.75, gamma=2.0),
        'optimizer': {
            'type': 'AdamW',
            'lr': 3e-4,
            'weight_decay': 5e-4
        },
        'scheduler': {
            'type': 'ReduceLROnPlateau',
            'mode': 'min',
            'factor': 0.5,
            'patience': 6,
            'min_lr': 1e-6
        },
        'batch_size': 64,
        'epochs': 200,
        'early_stopping': {
            'patience': 15,
            'min_delta': 0.003
        }
    }


# ============ 方案2B：动态难样本挖掘 ============
class OnlineHardExampleMining(nn.Module):
    """在线难样本挖掘（OHEM）"""
    def __init__(self, keep_ratio=0.7):
        """
        Args:
            keep_ratio: 保留损失最大的样本比例
        """
        super(OnlineHardExampleMining, self).__init__()
        self.keep_ratio = keep_ratio
    
    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 选择损失最大的样本
        num_keep = int(batch_size * self.keep_ratio)
        sorted_loss, _ = torch.sort(ce_loss, descending=True)
        threshold = sorted_loss[num_keep - 1]
        
        # 只对难样本计算损失
        mask = (ce_loss >= threshold).float()
        loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)
        
        return loss


# ============ 集成三种Loss的训练函数 ============
def train_with_advanced_loss(train_loader, model, criterion, optimizer, 
                             epoch, args, history, device):
    """使用高级损失函数训练"""
    from tqdm import tqdm
    
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == targets.data)
        total_samples += images.size(0)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': running_loss / total_samples,
            'acc': (running_corrects.double() / total_samples).item()
        })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc.item())
    
    return epoch_loss, epoch_acc


# ============ 使用示例：修改main_worker函数 ============
def modified_main_worker_focal(gpu, ngpus_per_node, args):
    """
    在原代码基础上修改以下部分：
    """
    # ... 前面的model初始化代码不变 ...
    
    # ========== 关键修改1: 使用Focal Loss ==========
    criterion = FocalLoss(
        alpha=0.75,      # 溢出类权重
        gamma=2.0,       # 聚焦难样本
        reduction='mean'
    ).cuda(args.gpu)
    
    # ========== 关键修改2: 优化器配置 ==========
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,         # 比模型2更保守
        weight_decay=5e-4
    )
    
    # ========== 关键修改3: 学习率调度 ==========
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=6,
        min_lr=1e-6,
        verbose=True
    )
    
    # ... 后续训练循环不变 ...
    
    return model


# ============ 多Loss组合策略 ============
class CombinedLoss(nn.Module):
    """组合多种Loss"""
    def __init__(self, focal_weight=0.7, ce_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)
        self.ce = nn.CrossEntropyLoss()
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
    
    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        ce_loss = self.ce(inputs, targets)
        return self.focal_weight * focal_loss + self.ce_weight * ce_loss
