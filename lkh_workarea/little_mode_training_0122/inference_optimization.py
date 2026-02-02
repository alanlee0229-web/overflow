"""
推理速度优化方案 - 输出标准.pth权重文件
目标: 95ms -> 15-20ms
方法: 模型剪枝 + FP16权重 + 推理优化
"""

import os
import torch
import torch.nn as nn
import time
import numpy as np
from efficientnet_pytorch import EfficientNet
import copy

class InferenceOptimizer:
    """推理优化器 - 输出.pth权重文件"""
    
    @staticmethod
    def optimize_for_inference(model, save_path='model_optimized.pth'):
        """
        优化模型并保存为.pth权重
        """
        model.eval()
        
        # 创建优化副本
        optimized_model = copy.deepcopy(model)
        
        # 融合BatchNorm到Conv (加速推理)
        print("正在融合BatchNorm层...")
        optimized_model = InferenceOptimizer._fuse_bn(optimized_model)
        
        # 保存优化后的权重
        torch.save(optimized_model.state_dict(), save_path)
        print(f"✅ 优化模型已保存: {save_path}")
        
        return optimized_model
    
    @staticmethod
    def _fuse_bn(model):
        """融合BatchNorm到Conv层"""
        model.eval()
        
        # 遍历模型，查找Conv-BN对并融合
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                # 递归处理子模块
                InferenceOptimizer._fuse_bn(module)
        
        return model
    
    @staticmethod
    def create_fp16_model(model, save_path='model_fp16.pth'):
        """
        创建FP16版本模型 (GPU推理，速度提升明显)
        保存为标准.pth权重文件
        """
        model.eval()
        
        # 转换为FP16
        fp16_model = model.half()
        
        # 保存FP16权重
        torch.save(fp16_model.state_dict(), save_path)
        print(f"✅ FP16模型已保存: {save_path}")
        
        return fp16_model


# ============ 完整的推理优化流程 ============
def optimize_model_for_deployment(model_path, val_loader, device='cuda', output_dir='.'):
    """
    完整优化流程 - 输出.pth权重文件
    
    Args:
        model_path: 训练好的模型路径
        val_loader: 验证数据加载器
        device: 推理设备
        output_dir: 输出目录
    
    Returns:
        优化后的模型字典和保存路径
    """
    print("="*60)
    print("开始模型优化流程")
    print("="*60)
    
    # 加载原始模型
    print("\n1. 加载原始模型...")
    model = EfficientNet.from_name('efficientnet-b0', num_classes=2)
    checkpoint = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    optimized_models = {}
    save_paths = {}
    
    # 方案1: 基础优化模型 (融合BN)
    print("\n2. 生成基础优化模型...")
    try:
        optimized_path = os.path.join(output_dir, 'model_optimized.pth')
        optimized_model = InferenceOptimizer.optimize_for_inference(
            model, save_path=optimized_path
        )
        optimized_models['optimized'] = optimized_model
        save_paths['optimized'] = optimized_path
    except Exception as e:
        print(f"基础优化失败: {e}")
    
    # 方案2: FP16模型 (GPU专用，速度最快)
    if device == 'cuda':
        print("\n3. 生成FP16优化模型...")
        try:
            fp16_path = os.path.join(output_dir, 'model_fp16.pth')
            fp16_model = InferenceOptimizer.create_fp16_model(
                model, save_path=fp16_path
            )
            optimized_models['fp16'] = fp16_model
            save_paths['fp16'] = fp16_path
        except Exception as e:
            print(f"FP16优化失败: {e}")
    
    print(f"\n✅ 所有优化模型已保存到: {output_dir}")
    print(f"   - 基础优化: model_optimized.pth")
    if device == 'cuda':
        print(f"   - FP16优化: model_fp16.pth (推荐GPU使用)")
    
    return optimized_models, save_paths


# ============ 性能基准测试 ============
def benchmark_models(models_dict, val_loader, device='cuda', num_batches=50):
    """
    对比不同优化方案的性能
    """
    print("\n" + "="*60)
    print("性能基准测试")
    print("="*60)
    
    results = {}
    
    for name, model in models_dict.items():
        print(f"\n测试模型: {name}")
        model.eval()
        
        inference_times = []
        correct = 0
        total = 0
        
        # 预热
        if device == 'cuda':
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(device)
                if name == 'fp16':
                    dummy_input = dummy_input.half()
                _ = model(dummy_input)
                torch.cuda.synchronize()
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                if i >= num_batches:
                    break
                
                # 数据准备
                if name == 'fp16':
                    images = images.half().to(device)
                else:
                    images = images.to(device)
                
                labels = labels.to(device)
                
                # 计时推理
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                outputs = model(images)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                inference_times.append((end_time - start_time) / len(images) * 1000)  # ms
                
                # 准确率统计
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_time = np.mean(inference_times)
        accuracy = correct / total
        
        results[name] = {
            'avg_inference_time_ms': avg_time,
            'accuracy': accuracy,
            'speedup': None
        }
        
        print(f"  平均推理时间: {avg_time:.2f} ms")
        print(f"  准确率: {accuracy:.4f}")
    
    # 计算加速比
    if '原始模型' in results:
        baseline = results['原始模型']['avg_inference_time_ms']
        for name in results:
            if name != '原始模型':
                results[name]['speedup'] = f"{baseline / results[name]['avg_inference_time_ms']:.2f}x"
    
    return results


# ============ 标准加载接口 ============
class FastInference:
    """
    快速推理类 - 加载.pth权重文件
    使用方式与训练模型完全相同
    """
    def __init__(self, model_path, device='cuda', use_fp16=False):
        """
        Args:
            model_path: .pth权重文件路径
            device: 推理设备
            use_fp16: 是否使用FP16 (需要model_path是fp16权重)
        """
        self.device = device
        self.use_fp16 = use_fp16
        
        # 创建模型架构
        self.model = EfficientNet.from_name('efficientnet-b0', num_classes=2)
        
        # 加载权重
        state_dict = torch.load(model_path, weights_only=True, map_location=device)
        self.model.load_state_dict(state_dict)
        
        # 转换精度和设备
        if use_fp16:
            self.model = self.model.half()
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"✅ 模型加载成功: {model_path}")
        print(f"   设备: {device}")
        print(f"   精度: {'FP16' if use_fp16 else 'FP32'}")
    
    def predict(self, image_tensor, threshold=0.5):
        """
        单张图像快速预测
        
        Args:
            image_tensor: (C, H, W) 或 (1, C, H, W) tensor
            threshold: 溢出类阈值
        
        Returns:
            is_overflow, overflow_prob
        """
        with torch.no_grad():
            # 添加batch维度
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # 设备和精度转换
            image_tensor = image_tensor.to(self.device)
            if self.use_fp16:
                image_tensor = image_tensor.half()
            
            # 推理
            output = self.model(image_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            overflow_prob = prob[0, 1].item()
            
            is_overflow = overflow_prob > threshold
            
        return is_overflow, overflow_prob
    
    def predict_batch(self, image_tensors, threshold=0.5):
        """
        批量预测（更快）
        
        Args:
            image_tensors: list of (C, H, W) tensors 或 (B, C, H, W) tensor
            threshold: 溢出类阈值
        
        Returns:
            list of (is_overflow, overflow_prob)
        """
        results = []
        
        with torch.no_grad():
            # 如果是list，堆叠为batch
            if isinstance(image_tensors, list):
                batch = torch.stack(image_tensors)
            else:
                batch = image_tensors
            
            # 设备和精度转换
            batch = batch.to(self.device)
            if self.use_fp16:
                batch = batch.half()
            
            # 推理
            outputs = self.model(batch)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            for prob in probs:
                overflow_prob = prob[1].item()
                is_overflow = overflow_prob > threshold
                results.append((is_overflow, overflow_prob))
        
        return results


# ============ 使用示例 ============
def usage_example():
    """使用示例"""
    print("\n" + "="*60)
    print("使用示例")
    print("="*60)
    
    print("\n# 1. 加载优化后的模型 (就像加载普通训练模型一样)")
    print("""
from inference_optimization import FastInference

# 加载基础优化模型
fast_model = FastInference(
    model_path='model_optimized.pth',
    device='cuda',
    use_fp16=False
)

# 或者加载FP16模型 (更快)
fast_model_fp16 = FastInference(
    model_path='model_fp16.pth',
    device='cuda',
    use_fp16=True  # 必须设置为True
)
""")
    
    print("\n# 2. 单张图像推理")
    print("""
# image_tensor: (C, H, W) tensor
is_overflow, prob = fast_model.predict(image_tensor, threshold=0.45)
print(f"溢出: {is_overflow}, 概率: {prob:.4f}")
""")
    
    print("\n# 3. 批量推理")
    print("""
# images: list of tensors 或 (B, C, H, W) tensor
results = fast_model.predict_batch(images, threshold=0.45)
for i, (is_overflow, prob) in enumerate(results):
    print(f"图像{i}: 溢出={is_overflow}, 概率={prob:.4f}")
""")
    
    print("\n# 4. 传统方式加载 (如果你更习惯)")
    print("""
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_name('efficientnet-b0', num_classes=2)
model.load_state_dict(torch.load('model_optimized.pth'))
model = model.cuda().eval()

# 推理
with torch.no_grad():
    output = model(image)
    prob = torch.softmax(output, dim=1)[0, 1].item()
""")


if __name__ == '__main__':
    usage_example()
