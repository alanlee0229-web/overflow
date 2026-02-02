"""
召回率99%+ & 精确率95%+ 双优化方案
当前: Recall=98.88%, Precision=92.66%
目标: Recall≥99%, Precision≥95%

策略: 阈值优化 + TTA + 后处理
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

class ThresholdOptimizer:
    """阈值优化器 - 在召回率约束下最大化精确率"""
    
    @staticmethod
    def find_optimal_threshold_constrained(model, val_loader, device, 
                                          min_recall=0.99, plot=True):
        """
        在保证最低召回率的前提下，找到最优阈值
        
        Args:
            min_recall: 最低召回率要求（默认99%）
        
        Returns:
            optimal_threshold, metrics
        """
        print(f"\n寻找最优阈值 (召回率 ≥ {min_recall*100}%)")
        print("="*60)
        
        model.eval()
        all_probs = []
        all_labels = []
        
        # 收集所有预测概率
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # 计算PR曲线
        precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
        
        # 找到满足召回率要求的最优阈值
        valid_indices = recall >= min_recall
        
        if not np.any(valid_indices):
            print(f"⚠️ 警告: 无法达到{min_recall*100}%召回率")
            # 退而求其次，找到最高召回率
            best_idx = np.argmax(recall)
        else:
            # 在满足召回率的前提下，选择精确率最高的
            valid_precision = precision[valid_indices]
            valid_recall = recall[valid_indices]
            valid_thresholds = thresholds[valid_indices[:-1]]  # thresholds比precision少1
            
            best_idx = np.argmax(valid_precision)
            optimal_threshold = valid_thresholds[best_idx]
            optimal_precision = valid_precision[best_idx]
            optimal_recall = valid_recall[best_idx]
        
        # 如果找不到合适的阈值，使用默认值
        if 'optimal_threshold' not in locals():
            optimal_threshold = 0.5
            preds = (all_probs > optimal_threshold).astype(int)
            from sklearn.metrics import precision_score, recall_score
            optimal_precision = precision_score(all_labels, preds)
            optimal_recall = recall_score(all_labels, preds)
        
        # 计算F1
        preds = (all_probs > optimal_threshold).astype(int)
        optimal_f1 = f1_score(all_labels, preds)
        
        print(f"\n✅ 最优阈值: {optimal_threshold:.4f}")
        print(f"   召回率: {optimal_recall:.4f} ({optimal_recall*100:.2f}%)")
        print(f"   精确率: {optimal_precision:.4f} ({optimal_precision*100:.2f}%)")
        print(f"   F1分数: {optimal_f1:.4f}")
        
        # 可视化
        if plot:
            ThresholdOptimizer._plot_pr_curve(
                precision, recall, thresholds,
                optimal_threshold, optimal_precision, optimal_recall
            )
        
        metrics = {
            'threshold': optimal_threshold,
            'precision': optimal_precision,
            'recall': optimal_recall,
            'f1': optimal_f1
        }
        
        return optimal_threshold, metrics
    
    @staticmethod
    def _plot_pr_curve(precision, recall, thresholds, 
                      opt_threshold, opt_precision, opt_recall):
        """绘制PR曲线"""
        plt.figure(figsize=(12, 5))
        
        # PR曲线
        plt.subplot(1, 2, 1)
        plt.plot(recall, precision, 'b-', linewidth=2, label='PR Curve')
        plt.plot(opt_recall, opt_precision, 'r*', markersize=15, 
                label=f'Optimal (T={opt_threshold:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 阈值 vs 指标
        plt.subplot(1, 2, 2)
        plt.plot(thresholds, precision[:-1], 'b-', label='Precision', linewidth=2)
        plt.plot(thresholds, recall[:-1], 'r-', label='Recall', linewidth=2)
        plt.axvline(opt_threshold, color='g', linestyle='--', 
                   label=f'Optimal T={opt_threshold:.3f}')
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Threshold vs Metrics', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('threshold_optimization.png', dpi=300)
        print("\n📊 阈值优化曲线已保存: threshold_optimization.png")


class AdvancedTTA:
    """高级TTA策略 - 提升召回率同时保持精确率"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def predict_with_tta(self, image_tensor, threshold=0.5, 
                        num_augmentations=5, aggregation='soft_vote'):
        """
        TTA预测
        
        Args:
            aggregation: 
                - 'soft_vote': 概率平均（推荐，平衡）
                - 'hard_vote': 多数投票（高召回）
                - 'conservative': 保守策略（高精确）
        """
        import torchvision.transforms.functional as TF
        
        self.model.eval()
        
        augmented_probs = []
        
        with torch.no_grad():
            # 原图
            img = image_tensor.unsqueeze(0).to(self.device)
            output = self.model(img)
            prob = F.softmax(output, dim=1)[0, 1].item()
            augmented_probs.append(prob)
            
            # TTA增强
            for _ in range(num_augmentations - 1):
                # 随机增强
                aug_img = image_tensor.clone()
                
                # 水平翻转 (50%概率)
                if np.random.rand() > 0.5:
                    aug_img = TF.hflip(aug_img)
                
                # 亮度调整
                brightness_factor = np.random.uniform(0.9, 1.1)
                aug_img = TF.adjust_brightness(aug_img, brightness_factor)
                
                # 对比度调整
                contrast_factor = np.random.uniform(0.9, 1.1)
                aug_img = TF.adjust_contrast(aug_img, contrast_factor)
                
                # 推理
                aug_img = aug_img.unsqueeze(0).to(self.device)
                output = self.model(aug_img)
                prob = F.softmax(output, dim=1)[0, 1].item()
                augmented_probs.append(prob)
        
        # 聚合策略
        if aggregation == 'soft_vote':
            final_prob = np.mean(augmented_probs)
            is_overflow = final_prob > threshold
        
        elif aggregation == 'hard_vote':
            votes = [p > threshold for p in augmented_probs]
            is_overflow = sum(votes) > len(votes) / 2
            final_prob = np.mean(augmented_probs)
        
        elif aggregation == 'conservative':
            # 保守策略：至少75%的增强认为是溢出
            votes = [p > threshold for p in augmented_probs]
            is_overflow = sum(votes) >= len(votes) * 0.75
            final_prob = np.mean(augmented_probs)
        
        return is_overflow, final_prob, augmented_probs


class PostProcessor:
    """后处理模块 - 时序平滑 + 规则过滤"""
    
    def __init__(self, window_size=5, min_detections=2):
        """
        Args:
            window_size: 时间窗口大小
            min_detections: 窗口内最少检测次数
        """
        self.window_size = window_size
        self.min_detections = min_detections
        self.history = []
        self.prob_history = []
    
    def update(self, is_overflow, overflow_prob):
        """
        更新检测历史并返回最终判断
        
        Returns:
            final_alarm: 最终是否报警
            confidence: 置信度
        """
        self.history.append(int(is_overflow))
        self.prob_history.append(overflow_prob)
        
        # 保持窗口大小
        if len(self.history) > self.window_size:
            self.history.pop(0)
            self.prob_history.pop(0)
        
        # 时序平滑判断
        detection_count = sum(self.history)
        avg_prob = np.mean(self.prob_history)
        
        # 触发条件
        final_alarm = detection_count >= self.min_detections
        
        # 置信度计算
        confidence = avg_prob if final_alarm else 1 - avg_prob
        
        return final_alarm, confidence
    
    def reset(self):
        """重置历史"""
        self.history = []
        self.prob_history = []


# ============ 完整评估流程 ============
def comprehensive_evaluation(model, val_loader, device):
    """
    综合评估 - 测试多种优化组合
    """
    print("\n" + "="*60)
    print("综合评估 - 测试多种优化策略")
    print("="*60)
    
    # 1. 基线（无优化）
    print("\n【基线测试】")
    baseline_metrics = evaluate_baseline(model, val_loader, device)
    
    # 2. 阈值优化
    print("\n【阈值优化】")
    optimal_threshold, threshold_metrics = ThresholdOptimizer.find_optimal_threshold_constrained(
        model, val_loader, device, min_recall=0.99, plot=True
    )
    
    # 3. TTA优化
    print("\n【TTA优化】")
    tta_metrics = evaluate_with_tta(
        model, val_loader, device, threshold=optimal_threshold
    )
    
    # 4. 时序平滑
    print("\n【时序平滑】")
    temporal_metrics = evaluate_with_temporal_smoothing(
        model, val_loader, device, threshold=optimal_threshold
    )
    
    # 汇总对比
    print("\n" + "="*60)
    print("性能对比汇总")
    print("="*60)
    
    comparison = {
        '基线': baseline_metrics,
        '阈值优化': threshold_metrics,
        'TTA优化': tta_metrics,
        '时序平滑': temporal_metrics
    }
    
    print(f"\n{'策略':<12} {'召回率':<10} {'精确率':<10} {'F1分数':<10}")
    print("-" * 50)
    for name, metrics in comparison.items():
        print(f"{name:<12} {metrics['recall']:<10.4f} {metrics['precision']:<10.4f} {metrics['f1']:<10.4f}")
    
    return comparison


def evaluate_baseline(model, val_loader, device, threshold=0.5):
    """基线评估"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = (probs[:, 1] > threshold).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    return {
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds)
    }


def evaluate_with_tta(model, val_loader, device, threshold=0.5):
    """TTA评估"""
    tta = AdvancedTTA(model, device)
    all_preds = []
    all_labels = []
    
    from tqdm import tqdm
    for images, labels in tqdm(val_loader, desc='TTA评估'):
        for i in range(len(images)):
            img = images[i]
            is_overflow, _, _ = tta.predict_with_tta(
                img, threshold=threshold, 
                num_augmentations=5, 
                aggregation='soft_vote'
            )
            all_preds.append(int(is_overflow))
            all_labels.append(labels[i].item())
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    return {
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds)
    }


def evaluate_with_temporal_smoothing(model, val_loader, device, threshold=0.5):
    """时序平滑评估（模拟视频流）"""
    model.eval()
    post_processor = PostProcessor(window_size=5, min_detections=2)
    
    all_final_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            for i in range(len(images)):
                overflow_prob = probs[i, 1].item()
                is_overflow = overflow_prob > threshold
                
                final_alarm, _ = post_processor.update(is_overflow, overflow_prob)
                
                all_final_preds.append(int(final_alarm))
                all_labels.append(labels[i].item())
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    return {
        'precision': precision_score(all_labels, all_final_preds),
        'recall': recall_score(all_labels, all_final_preds),
        'f1': f1_score(all_labels, all_final_preds)
    }


# ============ 实际部署类 - 集成所有优化 ============
class ProductionInference:
    """
    生产环境推理类
    集成: 阈值优化 + TTA + 时序平滑
    """
    def __init__(self, model, device='cuda', config=None):
        """
        Args:
            config: {
                'threshold': 0.45,
                'use_tta': True,
                'tta_num': 5,
                'temporal_window': 5,
                'temporal_min_detections': 2
            }
        """
        self.model = model
        self.device = device
        
        # 默认配置
        default_config = {
            'threshold': 0.45,
            'use_tta': True,
            'tta_num': 3,  # 实际部署建议3-5次
            'temporal_window': 5,
            'temporal_min_detections': 2
        }
        
        self.config = {**default_config, **(config or {})}
        
        # 初始化组件
        self.tta = AdvancedTTA(model, device) if self.config['use_tta'] else None
        self.post_processor = PostProcessor(
            window_size=self.config['temporal_window'],
            min_detections=self.config['temporal_min_detections']
        )
    
    def predict_frame(self, image_tensor):
        """
        单帧预测（视频流场景）
        
        Returns:
            {
                'instant_detection': bool,
                'final_alarm': bool,
                'confidence': float,
                'overflow_prob': float
            }
        """
        # TTA预测
        if self.config['use_tta']:
            is_overflow, overflow_prob, _ = self.tta.predict_with_tta(
                image_tensor,
                threshold=self.config['threshold'],
                num_augmentations=self.config['tta_num']
            )
        else:
            # 直接推理
            self.model.eval()
            with torch.no_grad():
                img = image_tensor.unsqueeze(0).to(self.device)
                output = self.model(img)
                prob = F.softmax(output, dim=1)[0, 1].item()
                is_overflow = prob > self.config['threshold']
                overflow_prob = prob
        
        # 时序平滑
        final_alarm, confidence = self.post_processor.update(is_overflow, overflow_prob)
        
        return {
            'instant_detection': is_overflow,
            'final_alarm': final_alarm,
            'confidence': confidence,
            'overflow_prob': overflow_prob
        }
    
    def reset(self):
        """重置时序历史（新视频开始时调用）"""
        self.post_processor.reset()
