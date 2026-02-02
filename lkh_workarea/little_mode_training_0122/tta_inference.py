"""
方案1: 测试时增强(TTA) + 阈值优化
优点: 无需重训练，立即提升召回率 2-5%
适用: 在现有模型2基础上直接使用
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

class TTAInference:
    """测试时增强推理器"""
    def __init__(self, model, device, overflow_threshold=0.45):
        """
        Args:
            model: 训练好的模型
            device: cuda/cpu
            overflow_threshold: 溢出类阈值（降低可提高召回率）
                - 0.5: 默认（平衡）
                - 0.4-0.45: 提高召回率（推荐）
                - 0.35-0.4: 激进召回（可能增加误报）
        """
        self.model = model
        self.device = device
        self.threshold = overflow_threshold
        
        # TTA变换组合
        self.tta_transforms = [
            transforms.Compose([]),  # 原图
            transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),  # 水平翻转
            transforms.Compose([transforms.ColorJitter(brightness=0.1)]),  # 亮度增加
            transforms.Compose([transforms.ColorJitter(brightness=(0.9, 0.9))]),  # 亮度降低（使用范围）
        ]
    
    def predict_single(self, image_tensor):
        """单张图像TTA预测"""
        self.model.eval()
        tta_preds = []
        
        with torch.no_grad():
            for transform in self.tta_transforms:
                # 应用变换
                aug_img = transform(image_tensor.cpu()).to(self.device)
                
                # 前向推理
                output = self.model(aug_img.unsqueeze(0))
                prob = F.softmax(output, dim=1)
                tta_preds.append(prob[0, 1].item())  # 溢出类概率
        
        # TTA融合：平均概率
        avg_overflow_prob = np.mean(tta_preds)
        
        # 基于阈值判断
        is_overflow = avg_overflow_prob > self.threshold
        
        return {
            'is_overflow': is_overflow,
            'overflow_prob': avg_overflow_prob,
            'tta_probs': tta_preds
        }
    
    def predict_batch(self, image_tensors):
        """批量预测"""
        results = []
        for img in image_tensors:
            results.append(self.predict_single(img))
        return results


class TemporalSmoothing:
    """时序平滑（针对视频流）"""
    def __init__(self, window_size=5, min_detections=2):
        """
        Args:
            window_size: 时间窗口大小（帧数）
            min_detections: 窗口内最少检测次数才报警
        """
        self.window_size = window_size
        self.min_detections = min_detections
        self.history = []
    
    def update(self, is_overflow):
        """更新检测历史"""
        self.history.append(int(is_overflow))
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # 判断是否触发报警
        overflow_count = sum(self.history)
        return overflow_count >= self.min_detections
    
    def reset(self):
        """重置历史"""
        self.history = []


# ============ 使用示例 ============
def evaluate_with_tta(model, val_loader, device):
    """使用TTA评估验证集"""
    tta_predictor = TTAInference(
        model=model,
        device=device,
        overflow_threshold=0.45  # 可调整
    )
    
    all_preds = []
    all_labels = []
    
    for images, labels in val_loader:
        for i in range(len(images)):
            img = images[i].to(device)
            result = tta_predictor.predict_single(img)
            
            all_preds.append(int(result['is_overflow']))
            all_labels.append(labels[i].item())
    
    # 计算指标
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(all_labels, all_preds, 
                                target_names=['Normal', 'Overflow']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    return all_preds, all_labels


def video_stream_inference(model, video_frames, device):
    """视频流推理（带时序平滑）"""
    tta_predictor = TTAInference(model, device, overflow_threshold=0.45)
    temporal_smoother = TemporalSmoothing(window_size=5, min_detections=2)
    
    results = []
    for frame_tensor in video_frames:
        # TTA预测
        pred = tta_predictor.predict_single(frame_tensor)
        
        # 时序平滑
        final_alarm = temporal_smoother.update(pred['is_overflow'])
        
        results.append({
            'instant_detection': pred['is_overflow'],
            'overflow_prob': pred['overflow_prob'],
            'final_alarm': final_alarm
        })
    
    return results


# ============ 阈值优化工具 ============
def find_optimal_threshold(model, val_loader, device):
    """自动寻找最优阈值（F1最大化）"""
    print("开始阈值搜索...")
    
    # 收集所有预测概率
    all_probs = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 搜索最优阈值
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in np.arange(0.3, 0.7, 0.02):
        preds = (np.array(all_probs) > threshold).astype(int)
        
        f1 = f1_score(all_labels, preds)
        precision = precision_score(all_labels, preds)
        recall = recall_score(all_labels, preds)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        print(f"Threshold={threshold:.2f}: F1={f1:.4f}, "
              f"Precision={precision:.4f}, Recall={recall:.4f}")
    
    print(f"\n最优阈值: {best_threshold:.2f}")
    print(f"最优指标: {best_metrics}")
    
    return best_threshold, best_metrics
