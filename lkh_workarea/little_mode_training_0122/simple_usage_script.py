"""
一键优化脚本 - 输出标准.pth权重文件
运行后得到:
  1. model_optimized.pth - 基础优化版本
  2. model_fp16.pth - FP16优化版本 (GPU推荐)
  3. threshold_optimization.pt - 最优阈值配置
  4. deployment_config.json - 部署配置文件

使用方式:
  python simple_usage_script.py
"""

import torch
import os
from pathlib import Path

# ==================== 配置区 ====================
# 修改这里的路径即可
MODEL_PATH = r"F:\work_area\___overflow\code_\mod_2_old\little_model\0122\tar\model_best_f1_0.9621.pth.tar"
VAL_DIR = r"F:\work_area\___overflow\pot_dataset\val"
OUTPUT_DIR = r"F:\work_area\___overflow\code_\mod_2_old\little_model\0122\jinghua"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MIN_RECALL = 0.99  # 最低召回率要求
# ================================================

def main():
    print("="*80)
    print("锅溢出检测模型 - 一键优化")
    print("="*80)
    print(f"\n配置:")
    print(f"  原始模型: {MODEL_PATH}")
    print(f"  验证集: {VAL_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  设备: {DEVICE}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ========== Step 1: 加载模型 ==========
    print("\n" + "="*60)
    print("Step 1: 加载原始模型")
    print("="*60)
    
    from efficientnet_pytorch import EfficientNet
    
    model = EfficientNet.from_name('efficientnet-b0', num_classes=2)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print("✅ 模型加载成功")
    
    # ========== Step 2: 加载验证集 ==========
    print("\n" + "="*60)
    print("Step 2: 加载验证集")
    print("="*60)
    
    from torchvision import transforms, datasets
    import PIL
    
    image_size = 224
    normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1, 1, 1])
    
    val_transforms = transforms.Compose([
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_dataset = datasets.ImageFolder(VAL_DIR, val_transforms)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✅ 验证集加载成功 (样本数: {len(val_dataset)})")
    
    # ========== Step 3: 阈值优化 ==========
    print("\n" + "="*60)
    print("Step 3: 阈值优化 (寻找最优分类阈值)")
    print("="*60)
    
    from precision_recall_optimizer import ThresholdOptimizer
    
    optimal_threshold, threshold_metrics = ThresholdOptimizer.find_optimal_threshold_constrained(
        model=model,
        val_loader=val_loader,
        device=DEVICE,
        min_recall=MIN_RECALL,
        plot=True
    )
    
    # 保存阈值结果
    threshold_result = {
        'optimal_threshold': float(optimal_threshold),
        'metrics': {k: float(v) for k, v in threshold_metrics.items()}
    }
    torch.save(threshold_result, os.path.join(OUTPUT_DIR, 'threshold_optimization.pt'))
    
    print(f"\n✅ 阈值优化完成")
    print(f"   最优阈值: {optimal_threshold:.4f}")
    print(f"   召回率: {threshold_metrics['recall']*100:.2f}%")
    print(f"   精确率: {threshold_metrics['precision']*100:.2f}%")
    print(f"   F1分数: {threshold_metrics['f1']*100:.2f}%")
    
    # ========== Step 4: 生成优化模型 (.pth权重) ==========
    print("\n" + "="*60)
    print("Step 4: 生成优化模型权重")
    print("="*60)
    
    from inference_optimization import optimize_model_for_deployment
    
    optimized_models, save_paths = optimize_model_for_deployment(
        model_path=MODEL_PATH,
        val_loader=val_loader,
        device=DEVICE,
        output_dir=OUTPUT_DIR
    )
    
    # ========== Step 5: 性能测试 ==========
    print("\n" + "="*60)
    print("Step 5: 性能基准测试")
    print("="*60)
    
    from inference_optimization import benchmark_models
    
    test_models = {
        '原始模型': model,
        **optimized_models
    }
    
    speed_results = benchmark_models(
        models_dict=test_models,
        val_loader=val_loader,
        device=DEVICE,
        num_batches=50
    )
    
    # 打印对比
    print("\n" + "="*60)
    print("性能对比")
    print("="*60)
    print(f"{'模型类型':<15} {'推理时间(ms)':<15} {'准确率':<10} {'加速比':<10}")
    print("-" * 60)
    
    for name, metrics in speed_results.items():
        speedup = metrics.get('speedup', '-')
        print(f"{name:<15} {metrics['avg_inference_time_ms']:<15.2f} "
              f"{metrics['accuracy']:<10.4f} {speedup:<10}")
    
    # ========== Step 6: 生成部署配置 ==========
    print("\n" + "="*60)
    print("Step 6: 生成部署配置")
    print("="*60)
    
    # 选择最快的模型
    optimized_results = {k: v for k, v in speed_results.items() if k != '原始模型'}
    if optimized_results:
        best_model_name = min(optimized_results, key=lambda x: optimized_results[x]['avg_inference_time_ms'])
        best_model_path = save_paths.get(best_model_name, '')
    else:
        best_model_name = '原始模型'
        best_model_path = MODEL_PATH
    
    deployment_config = {
        'model_type': best_model_name,
        'model_path': best_model_path,
        'threshold': float(optimal_threshold),
        'use_fp16': best_model_name == 'fp16',
        'device': DEVICE,
        'target_recall': MIN_RECALL,
        'achieved_metrics': {
            'recall': float(threshold_metrics['recall']),
            'precision': float(threshold_metrics['precision']),
            'f1': float(threshold_metrics['f1'])
        },
        'inference_time_ms': float(speed_results[best_model_name]['avg_inference_time_ms'])
    }
    
    import json
    config_path = os.path.join(OUTPUT_DIR, 'deployment_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(deployment_config, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ 部署配置已保存: {config_path}")
    print("\n配置内容:")
    print(json.dumps(deployment_config, indent=4, ensure_ascii=False))
    
    # ========== 最终总结 ==========
    print("\n" + "="*80)
    print("优化完成总结")
    print("="*80)
    
    print(f"\n📊 性能提升:")
    original_metrics = speed_results['原始模型']
    best_metrics = speed_results[best_model_name]
    
    print(f"  推理速度: {original_metrics['avg_inference_time_ms']:.2f}ms → {best_metrics['avg_inference_time_ms']:.2f}ms")
    print(f"  加速比: {best_metrics.get('speedup', 'N/A')}")
    print(f"  准确率: {best_metrics['accuracy']*100:.2f}%")
    
    print(f"\n📊 分类性能:")
    print(f"  召回率: {threshold_metrics['recall']*100:.2f}%")
    print(f"  精确率: {threshold_metrics['precision']*100:.2f}%")
    print(f"  F1分数: {threshold_metrics['f1']*100:.2f}%")
    print(f"  最优阈值: {optimal_threshold:.4f}")
    
    print(f"\n📦 输出文件:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    ├── model_optimized.pth          # 基础优化模型")
    if DEVICE == 'cuda':
        print(f"    ├── model_fp16.pth               # FP16优化模型 (推荐)")
    print(f"    ├── threshold_optimization.pt    # 阈值优化结果")
    print(f"    ├── threshold_optimization.png   # 阈值曲线图")
    print(f"    └── deployment_config.json       # 部署配置")
    
    print("\n✅ 所有优化完成！")
    
    # ========== 使用说明 ==========
    print("\n" + "="*80)
    print("快速使用指南")
    print("="*80)
    
    print("\n方法1: 使用FastInference类 (推荐)")
    print("-" * 60)
    print(f"""
from inference_optimization import FastInference

# 加载优化模型
model = FastInference(
    model_path='{best_model_path}',
    device='{DEVICE}',
    use_fp16={best_model_name == 'fp16'}
)

# 推理
is_overflow, prob = model.predict(image_tensor, threshold={optimal_threshold:.4f})
print(f"溢出: {{is_overflow}}, 概率: {{prob:.4f}}")
""")
    
    print("\n方法2: 传统方式 (兼容性最好)")
    print("-" * 60)
    print(f"""
from efficientnet_pytorch import EfficientNet
import torch

model = EfficientNet.from_name('efficientnet-b0', num_classes=2)
model.load_state_dict(torch.load('{best_model_path}'))
model = model.{DEVICE}().eval()
{'model = model.half()  # FP16模式' if best_model_name == 'fp16' else ''}

with torch.no_grad():
    output = model(image)
    prob = torch.softmax(output, dim=1)[0, 1].item()
    is_overflow = prob > {optimal_threshold:.4f}
""")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
