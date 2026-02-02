"""
方案3: 多Patch区域感知检测
优点: 专门捕获锅边缘小面积溢出，大幅降低早期漏检
实现: 轻量级改造，复用现有模型
预期提升: Recall +3-6%（特别是早期溢出）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class MultiPatchDetector:
    """多区域Patch检测器"""
    def __init__(self, model, device, patch_strategy='edge_focus'):
        """
        Args:
            model: 训练好的分类模型
            device: cuda/cpu
            patch_strategy: 
                - 'edge_focus': 专注锅边缘（推荐）
                - 'grid': 网格划分
                - 'adaptive': 自适应重点区域
        """
        self.model = model
        self.device = device
        self.strategy = patch_strategy
    
    def extract_edge_patches(self, image_tensor, patch_size=112):
        """
        提取锅边缘区域Patch
        
        锅溢出通常发生在：
        - 上边缘（最常见）
        - 左/右边缘
        - 中心区域（参考）
        """
        _, _, h, w = image_tensor.shape
        
        patches = []
        positions = []
        
        # 1. 上边缘（重点）
        top_patch = image_tensor[:, :, :patch_size, :]
        patches.append(self._resize_patch(top_patch, patch_size))
        positions.append('top')
        
        # 2. 左边缘
        left_patch = image_tensor[:, :, :, :patch_size]
        patches.append(self._resize_patch(left_patch, patch_size))
        positions.append('left')
        
        # 3. 右边缘
        right_patch = image_tensor[:, :, :, -patch_size:]
        patches.append(self._resize_patch(right_patch, patch_size))
        positions.append('right')
        
        # 4. 中心区域（对照）
        center_y = h // 2
        center_x = w // 2
        half_patch = patch_size // 2
        center_patch = image_tensor[
            :, :,
            center_y - half_patch:center_y + half_patch,
            center_x - half_patch:center_x + half_patch
        ]
        patches.append(self._resize_patch(center_patch, patch_size))
        positions.append('center')
        
        return patches, positions
    
    def _resize_patch(self, patch, target_size):
        """调整Patch到模型输入尺寸"""
        return F.interpolate(
            patch,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )
    
    def predict(self, image_tensor, voting_strategy='any'):
        """
        多Patch预测
        
        Args:
            voting_strategy:
                - 'any': 任一Patch检测到溢出即报警（高召回）
                - 'majority': 多数投票（平衡）
                - 'weighted': 加权投票（边缘权重更高）
        """
        self.model.eval()
        
        # 提取Patch
        patches, positions = self.extract_edge_patches(image_tensor)
        
        # 对每个Patch推理
        patch_results = {}
        with torch.no_grad():
            for patch, pos in zip(patches, positions):
                output = self.model(patch)
                prob = F.softmax(output, dim=1)
                overflow_prob = prob[0, 1].item()
                
                patch_results[pos] = {
                    'overflow_prob': overflow_prob,
                    'is_overflow': overflow_prob > 0.5
                }
        
        # 投票决策
        final_decision = self._vote(patch_results, voting_strategy)
        
        return {
            'is_overflow': final_decision,
            'patch_results': patch_results
        }
    
    def _vote(self, patch_results, strategy):
        """投票策略"""
        if strategy == 'any':
            # 任一Patch检测到即报警（最高召回率）
            return any(r['is_overflow'] for r in patch_results.values())
        
        elif strategy == 'majority':
            # 多数投票
            overflow_count = sum(r['is_overflow'] for r in patch_results.values())
            return overflow_count > len(patch_results) / 2
        
        elif strategy == 'weighted':
            # 边缘权重更高
            weights = {
                'top': 0.4,      # 上边缘最重要
                'left': 0.25,
                'right': 0.25,
                'center': 0.1
            }
            
            weighted_prob = sum(
                weights[pos] * result['overflow_prob']
                for pos, result in patch_results.items()
            )
            
            return weighted_prob > 0.5
        
        return False


# ============ 完整评估流程 ============
def evaluate_with_patches(model, val_loader, device, strategy='any'):
    """使用多Patch策略评估"""
    detector = MultiPatchDetector(model, device)
    
    all_preds = []
    all_labels = []
    
    from tqdm import tqdm
    for images, labels in tqdm(val_loader, desc='Multi-Patch Evaluation'):
        for i in range(len(images)):
            img = images[i].unsqueeze(0).to(device)
            
            result = detector.predict(img, voting_strategy=strategy)
            
            all_preds.append(int(result['is_overflow']))
            all_labels.append(labels[i].item())
    
    # 计算指标
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\n{'='*60}")
    print(f"Multi-Patch Strategy: {strategy}")
    print('='*60)
    print(classification_report(all_labels, all_preds,
                                target_names=['Normal', 'Overflow'],
                                digits=4))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"FN={cm[1,0]}, TP={cm[1,1]}")
    
    return all_preds, all_labels


# ============ 方案3B：自适应注意力Patch ============
class AttentionPatchDetector:
    """基于注意力的自适应Patch检测"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
        # 注册GradCAM hook
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """注册梯度钩子"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # 假设使用EfficientNet，获取最后一个卷积层
        if hasattr(self.model, '_conv_head'):
            target_layer = self.model._conv_head
        else:
            # 如果是DataParallel包装的
            target_layer = self.model.module._conv_head
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def get_attention_map(self, image_tensor, target_class=1):
        """获取注意力热图（GradCAM）"""
        self.model.eval()
        
        # 前向传播
        output = self.model(image_tensor)
        
        # 反向传播
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # 计算GradCAM
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def extract_high_attention_patches(self, image_tensor, attention_map, 
                                      num_patches=3, patch_size=112):
        """基于注意力提取关键区域Patch"""
        import cv2
        
        # Resize attention map to image size
        _, _, h, w = image_tensor.shape
        attention_resized = cv2.resize(attention_map, (w, h))
        
        # 找到注意力最高的区域
        patches = []
        
        # 展平并找top-k位置
        flat_attention = attention_resized.flatten()
        top_indices = np.argsort(flat_attention)[-num_patches * 100:]
        
        for idx in top_indices[::100][:num_patches]:
            y = idx // w
            x = idx % w
            
            # 提取以该点为中心的patch
            half_patch = patch_size // 2
            y1 = max(0, y - half_patch)
            y2 = min(h, y + half_patch)
            x1 = max(0, x - half_patch)
            x2 = min(w, x + half_patch)
            
            patch = image_tensor[:, :, y1:y2, x1:x2]
            patch_resized = F.interpolate(
                patch,
                size=(patch_size, patch_size),
                mode='bilinear'
            )
            patches.append(patch_resized)
        
        return patches


# ============ 使用示例 ============
def compare_patch_strategies(model, val_loader, device):
    """对比不同Patch策略"""
    strategies = ['any', 'majority', 'weighted']
    
    results = {}
    for strategy in strategies:
        print(f"\n测试策略: {strategy}")
        preds, labels = evaluate_with_patches(model, val_loader, device, strategy)
        
        from sklearn.metrics import f1_score, precision_score, recall_score
        results[strategy] = {
            'f1': f1_score(labels, preds),
            'precision': precision_score(labels, preds),
            'recall': recall_score(labels, preds)
        }
    
    # 输出对比
    print("\n" + "="*60)
    print("策略对比汇总:")
    print("="*60)
    for strategy, metrics in results.items():
        print(f"{strategy:>10s}: F1={metrics['f1']:.4f}, "
              f"Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}")
    
    return results
