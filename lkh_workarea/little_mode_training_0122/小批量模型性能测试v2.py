import os
import cv2
import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import time

# 导入TTA相关类
from tta_inference import TTAInference, TemporalSmoothing

# 设置matplotlib字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置区域 ====================
# 验证文件夹根目录列表（包含0和1子文件夹）
VALIDATION_FOLDERS = [
    r"F:\work_area\___overflow\pot_dataset_vaild1a2b\val_2",
    r"F:\work_area\___overflow\pot_dataset_vaild1a2b\val_3",
]

# 分类模型路径列表（支持测试多个模型）
CLASSIFICATION_MODEL_PATHS = [
    r"F:\work_area\___overflow\code_\mod_2_old\little_model\0122\model_0112_datasetV11.2.pth",
    # r"F:\work_area\___overflow\code_\mod_2_old\little_model\0122\model_0112_jinghua.pt",
    r"F:\work_area\___overflow\code_\mod_2_old\little_model\0122\model_V13.1_model2_Optimized.pth",
    r"F:\work_area\___overflow\code_\lkh_py\little_mode_training\model_optimized.pth"
]

# CLASSIFICATION_MODEL_PATHS = [
#     r"F:\work_area\___overflow\code_\mod_2_old\little_model\0122\model_V13.1_model2_Focalloss.pth",
#     r"F:\work_area\___overflow\code_\mod_2_old\little_model\0122\model_V13.1_model2_Optimized.pth",
#     r"F:\work_area\___overflow\code_\mod_2_old\little_model\0122\model_V13.2_model3_Focalloss.pth",
#     r"F:\work_area\___overflow\code_\mod_2_old\little_model\0122\model_V13.2_model3_Optimized.pth"
# ]

# 结果报告输出路径
RESULT_OUTPUT_PATH = r"F:\work_area\___overflow\code_\mod_2_old\little_model\0122\chart_v2_2jinghua\validation_results.txt"
# 可视化图表输出目录
CHART_OUTPUT_DIR = r"F:\work_area\___overflow\code_\mod_2_old\little_model\0122\chart_v2_2jinghua"

# 结果报告输出路径
#RESULT_OUTPUT_PATH = r"F:\work_area\___overflow\code_\mod_2_old\little_model\0122\charts\validation_results.txt"
# 可视化图表输出目录
#CHART_OUTPUT_DIR = r"F:\work_area\___overflow\code_\mod_2_old\little_model\0122\charts"

# 图像输入尺寸
INPUT_SIZE = 224

# 模型类别配置
class_names = {
    0: 'no_boiling_over',
    1: 'boiling over',
}

# TTA配置
USE_TTA = True  # 是否使用测试时增强
OVERFLOW_THRESHOLD = 0.45  # 溢出阈值

# 阈值调优配置
TUNE_THRESHOLD = False  # 是否进行阈值调优
THRESHOLD_RANGE = (0.3, 0.7)  # 阈值搜索范围
THRESHOLD_STEP = 0.02  # 阈值搜索步长

class ModelValidator:
    def __init__(self, classification_model_paths):
        self.classification_model_paths = classification_model_paths
        self.classification_models = []  # 存储多个分类模型
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 创建图表输出目录
        os.makedirs(CHART_OUTPUT_DIR, exist_ok=True)
        
        self.load_models()
    
    def load_models(self):
        print("正在加载模型...")
        
        # 加载多个分类模型
        self.classification_models = []
        for i, model_path in enumerate(self.classification_model_paths):
            try:
                start_time = time.time()
                # 1. 加载模型
                classification_model = EfficientNet.from_name('efficientnet-b0', num_classes=2)
                
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    classification_model.load_state_dict(checkpoint['state_dict'])
                else:
                    classification_model.load_state_dict(checkpoint)
                classification_model.eval()
                # 将模型移到正确的设备
                classification_model = classification_model.to(self.device)
                
                # 2. 创建TTA推理器
                tta_inference = TTAInference(
                    model=classification_model,
                    device=self.device,
                    overflow_threshold=OVERFLOW_THRESHOLD
                )
                
                load_time = time.time() - start_time
                self.classification_models.append({
                    'model': classification_model,
                    'tta_inference': tta_inference,
                    'path': model_path,
                    'name': f"模型{i+1}",
                    'load_time': load_time
                })
                print(f"分类模型{i+1}加载成功: {model_path} (加载时间: {load_time:.2f}秒)")
            except Exception as e:
                print(f"分类模型{i+1}加载失败: {str(e)}")
                raise
        
        print("所有模型加载完成！\n")
    
    def predict_single_image(self, image_path):
        """对单张图像进行推理预测"""
        if not os.path.exists(image_path):
            return None, f"图像文件不存在: {image_path}", 0
        
        try:
            start_time = time.time()
            frame = cv2.imread(image_path)
            if frame is None:
                return None, f"无法读取图像: {image_path}", 0
            
            # 图像预处理
            # 将裁剪图直接拉伸为 224x224
            image = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            
            # 转换为RGB（训练时是RGB）
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 转换为float32并归一化到[0,1]
            image_float = image_rgb.astype(np.float32) / 255.0
            
            # 转换为tensor并调整为NCHW格式
            efficientnet_input = torch.from_numpy(image_float).permute(2, 0, 1)
            
            # 对每个分类模型进行推理
            model_results = []
            for model_info in self.classification_models:
                model = model_info['model']
                model_name = model_info['name']
                tta_inference = model_info['tta_inference']
                
                with torch.no_grad():
                    inference_start = time.time()
                    
                    if USE_TTA:
                        # 使用TTA推理
                        tta_result = tta_inference.predict_single(efficientnet_input)
                        overflow_prob = tta_result['overflow_prob']
                        predicted_class_index = 1 if overflow_prob > OVERFLOW_THRESHOLD else 0
                        confidence = overflow_prob if predicted_class_index == 1 else 1 - overflow_prob
                    else:
                        # 不使用TTA，直接推理
                        classification_results = model(efficientnet_input.unsqueeze(0))
                        probabilities = torch.softmax(classification_results, dim=1)
                        overflow_prob = probabilities[0, 1].item()
                        predicted_class_index = 1 if overflow_prob > OVERFLOW_THRESHOLD else 0
                        confidence = overflow_prob if predicted_class_index == 1 else 1 - overflow_prob
                    
                    predicted_class = class_names.get(predicted_class_index, f"未知类别({predicted_class_index})")
                    inference_time = time.time() - inference_start
                
                model_results.append({
                    'model_name': model_name,
                    'class_index': predicted_class_index,
                    'class_name': predicted_class,
                    'confidence': confidence,
                    'overflow_prob': overflow_prob,
                    'inference_time': inference_time
                })
            
            total_time = time.time() - start_time
            return model_results, None, total_time
        except Exception as e:
            return None, f"推理过程中出错: {str(e)}", 0
    
    def validate_folder(self, folder_path, folder_name):
        """验证单个文件夹（包含0和1子文件夹）"""
        print(f"\n{'='*60}")
        print(f"开始验证文件夹: {folder_name}")
        print(f"路径: {folder_path}")
        print(f"{'='*60}\n")
        
        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            return None
        
        results = {
            'folder_name': folder_name,
            'total_images': 0,
            'model_results': {model['name']: {'true_labels': [], 'predicted_labels': [], 'inference_times': [], 'overflow_probs': []} for model in self.classification_models},
            'details': []
        }
        
        # 验证0文件夹（未溢出）
        class_0_path = os.path.join(folder_path, "0")
        if os.path.exists(class_0_path):
            print(f"处理类别0文件夹（未溢出）: {class_0_path}")
            self.process_class_folder(class_0_path, true_class=0, results=results)
        else:
            print(f"警告: 类别0文件夹不存在: {class_0_path}")
        
        # 验证1文件夹（溢出）
        class_1_path = os.path.join(folder_path, "1")
        if os.path.exists(class_1_path):
            print(f"\n处理类别1文件夹（溢出）: {class_1_path}")
            self.process_class_folder(class_1_path, true_class=1, results=results)
        else:
            print(f"警告: 类别1文件夹不存在: {class_1_path}")
        
        return results
    
    def process_class_folder(self, class_folder_path, true_class, results):
        """处理单个类别文件夹"""
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for file in os.listdir(class_folder_path):
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(file)
        
        if not image_files:
            print(f"  警告: 文件夹中没有找到图像文件")
            return
        
        print(f"  共找到 {len(image_files)} 张图像")
        
        # 处理每张图像
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(class_folder_path, image_file)
            model_results, error, total_time = self.predict_single_image(image_path)
            
            if error:
                print(f"  [{i+1}/{len(image_files)}] 错误: {image_file} - {error}")
                continue
            
            results['total_images'] += 1
            
            # 记录每个模型的结果
            for model_result in model_results:
                model_name = model_result['model_name']
                results['model_results'][model_name]['true_labels'].append(true_class)
                results['model_results'][model_name]['predicted_labels'].append(model_result['class_index'])
                results['model_results'][model_name]['inference_times'].append(model_result['inference_time'])
                results['model_results'][model_name]['overflow_probs'].append(model_result['overflow_prob'])
            
            true_class_name = class_names.get(true_class, f"未知类别({true_class})")
            
            print(f"  [{i+1}/{len(image_files)}] {image_file}")
            for model_result in model_results:
                print(f"    {model_result['model_name']}: {model_result['class_name']} (置信度: {model_result['confidence']:.4f}, 推理时间: {model_result['inference_time']*1000:.2f}ms)")
            
            results['details'].append({
                'filename': image_file,
                'true_class': true_class,
                'true_class_name': true_class_name,
                'model_results': model_results
            })
    
    def calculate_metrics(self, true_labels, predicted_labels):
        """计算分类指标"""
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='binary', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='binary', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average='binary', zero_division=0)
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix
        }
    
    def find_optimal_threshold(self, true_labels, overflow_probs):
        """寻找最优阈值（F1最大化）"""
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        best_f1 = 0
        best_threshold = OVERFLOW_THRESHOLD
        best_metrics = {}
        
        for threshold in np.arange(THRESHOLD_RANGE[0], THRESHOLD_RANGE[1], THRESHOLD_STEP):
            preds = (np.array(overflow_probs) > threshold).astype(int)
            
            f1 = f1_score(true_labels, preds)
            precision = precision_score(true_labels, preds)
            recall = recall_score(true_labels, preds)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'f1': f1,
                    'precision': precision,
                    'recall': recall
                }
        
        return best_threshold, best_metrics
    
    def generate_model_statistics(self, results):
        """生成单个模型的统计结果"""
        model_stats = {}
        
        for model_name, model_result in results['model_results'].items():
            if not model_result['true_labels']:
                continue
            
            metrics = self.calculate_metrics(model_result['true_labels'], model_result['predicted_labels'])
            
            # 计算推理效率
            avg_inference_time = np.mean(model_result['inference_times'])
            total_inference_time = np.sum(model_result['inference_times'])
            
            # 阈值调优
            optimal_threshold = None
            optimal_metrics = None
            if TUNE_THRESHOLD and len(model_result['overflow_probs']) > 0:
                optimal_threshold, optimal_metrics = self.find_optimal_threshold(
                    model_result['true_labels'],
                    model_result['overflow_probs']
                )
            
            model_stats[model_name] = {
                'metrics': metrics,
                'inference_stats': {
                    'avg_inference_time': avg_inference_time,
                    'total_inference_time': total_inference_time,
                    'total_images': len(model_result['true_labels'])
                },
                'true_labels': model_result['true_labels'],
                'predicted_labels': model_result['predicted_labels'],
                'overflow_probs': model_result['overflow_probs'],
                'optimal_threshold': optimal_threshold,
                'optimal_metrics': optimal_metrics
            }
        
        return model_stats
    
    def print_summary(self, results):
        """打印验证结果摘要"""
        if results is None:
            print("没有可显示的验证结果")
            return
        
        model_stats = self.generate_model_statistics(results)
        
        print(f"\n{'='*60}")
        print(f"验证结果摘要: {results['folder_name']}")
        print(f"{'='*60}")
        print(f"总图像数: {results['total_images']}")
        print(f"使用TTA: {'是' if USE_TTA else '否'}")
        print(f"当前阈值: {OVERFLOW_THRESHOLD:.2f}")
        
        # 打印各模型的详细统计
        print(f"\n各模型详细统计:")
        for model_name, stats in model_stats.items():
            metrics = stats['metrics']
            inference_stats = stats['inference_stats']
            
            print(f"\n{model_name}:")
            print(f"  准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"  精确率: {metrics['precision']:.4f}")
            print(f"  召回率: {metrics['recall']:.4f}")
            print(f"  F1分数: {metrics['f1']:.4f}")
            print(f"  平均推理时间: {inference_stats['avg_inference_time']*1000:.2f}ms")
            print(f"  总推理时间: {inference_stats['total_inference_time']:.2f}s")
            
            # 打印混淆矩阵
            print(f"  混淆矩阵:")
            cm = metrics['confusion_matrix']
            print(f"    真实0 / 预测0: {cm[0][0]}")
            print(f"    真实0 / 预测1: {cm[0][1]}")
            print(f"    真实1 / 预测0: {cm[1][0]}")
            print(f"    真实1 / 预测1: {cm[1][1]}")
            
            # 打印最优阈值（如果有）
            if TUNE_THRESHOLD and stats['optimal_threshold'] is not None:
                print(f"  最优阈值: {stats['optimal_threshold']:.2f}")
                print(f"  最优F1分数: {stats['optimal_metrics']['f1']:.4f}")
                print(f"  最优精确率: {stats['optimal_metrics']['precision']:.4f}")
                print(f"  最优召回率: {stats['optimal_metrics']['recall']:.4f}")
        
        print(f"{'='*60}\n")
    
    def print_overall_summary(self, all_results):
        """打印总体验证结果摘要"""
        print(f"\n{'='*60}")
        print("总体验证结果摘要")
        print(f"{'='*60}")
        
        # 合并所有结果
        overall_results = {
            'total_images': 0,
            'model_results': {model['name']: {'true_labels': [], 'predicted_labels': [], 'inference_times': [], 'overflow_probs': []} for model in self.classification_models}
        }
        
        for results in all_results:
            if results is None:
                continue
            
            overall_results['total_images'] += results['total_images']
            
            for model_name, model_result in results['model_results'].items():
                overall_results['model_results'][model_name]['true_labels'].extend(model_result['true_labels'])
                overall_results['model_results'][model_name]['predicted_labels'].extend(model_result['predicted_labels'])
                overall_results['model_results'][model_name]['inference_times'].extend(model_result['inference_times'])
                overall_results['model_results'][model_name]['overflow_probs'].extend(model_result['overflow_probs'])
        
        print(f"总图像数: {overall_results['total_images']}")
        print(f"使用TTA: {'是' if USE_TTA else '否'}")
        print(f"当前阈值: {OVERFLOW_THRESHOLD:.2f}")
        
        # 计算各模型的总体统计
        overall_model_stats = self.generate_model_statistics(overall_results)
        
        # 打印各模型的总体统计
        print(f"\n各模型总体统计:")
        for model_name, stats in overall_model_stats.items():
            metrics = stats['metrics']
            inference_stats = stats['inference_stats']
            
            print(f"\n{model_name}:")
            print(f"  准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"  精确率: {metrics['precision']:.4f}")
            print(f"  召回率: {metrics['recall']:.4f}")
            print(f"  F1分数: {metrics['f1']:.4f}")
            print(f"  平均推理时间: {inference_stats['avg_inference_time']*1000:.2f}ms")
            print(f"  总推理时间: {inference_stats['total_inference_time']:.2f}s")
            
            # 打印最优阈值（如果有）
            if TUNE_THRESHOLD and stats['optimal_threshold'] is not None:
                print(f"  最优阈值: {stats['optimal_threshold']:.2f}")
                print(f"  最优F1分数: {stats['optimal_metrics']['f1']:.4f}")
                print(f"  最优精确率: {stats['optimal_metrics']['precision']:.4f}")
                print(f"  最优召回率: {stats['optimal_metrics']['recall']:.4f}")
        
        # 生成模型对比表格
        print(f"\n{'='*60}")
        print("模型性能对比")
        print(f"{'='*60}")
        
        # 打印对比表格
        print(f"{'模型名称':<10} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'平均推理时间(ms)':<20}")
        print(f"-{'-'*10}-{'-'*10}-{'-'*10}-{'-'*10}-{'-'*10}-{'-'*20}")
        
        for model_name, stats in overall_model_stats.items():
            metrics = stats['metrics']
            inference_stats = stats['inference_stats']
            print(f"{model_name:<10} {metrics['accuracy']*100:<9.2f}% {metrics['precision']:<9.4f} {metrics['recall']:<9.4f} {metrics['f1']:<9.4f} {inference_stats['avg_inference_time']*1000:<19.2f}")
        
        # 生成可视化图表
        self.generate_visualizations(overall_model_stats)
        
        print(f"\n{'='*60}\n")
    
    def generate_visualizations(self, overall_model_stats):
        """生成可视化图表"""
        # 1. 混淆矩阵对比
        self.plot_confusion_matrices(overall_model_stats)
        
        # 2. 性能指标对比
        self.plot_performance_metrics(overall_model_stats)
        
        # 3. 推理时间对比
        self.plot_inference_times(overall_model_stats)
    
    def plot_confusion_matrices(self, overall_model_stats):
        """绘制混淆矩阵对比图"""
        fig, axes = plt.subplots(1, len(overall_model_stats), figsize=(15, 5))
        
        for i, (model_name, stats) in enumerate(overall_model_stats.items()):
            cm = stats['metrics']['confusion_matrix']
            ax = axes[i] if len(overall_model_stats) > 1 else axes
            
            # 使用matplotlib绘制混淆矩阵
            cax = ax.matshow(cm, cmap='Blues')
            fig.colorbar(cax, ax=ax)
            
            # 添加数值标签
            for (j, k), val in np.ndenumerate(cm):
                ax.text(k, j, f'{val}', ha='center', va='center', color='white' if val > cm.max()/2 else 'black')
            
            ax.set_title(f'{model_name} 混淆矩阵')
            ax.set_xlabel('预测类别')
            ax.set_ylabel('真实类别')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['预测0', '预测1'])
            ax.set_yticklabels(['真实0', '真实1'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(CHART_OUTPUT_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"混淆矩阵对比图已保存: {os.path.join(CHART_OUTPUT_DIR, 'confusion_matrices.png')}")
    
    def plot_performance_metrics(self, overall_model_stats):
        """绘制性能指标对比图"""
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        model_names = list(overall_model_stats.keys())
        
        # 准备数据
        data = {}
        for metric in metrics:
            data[metric] = [overall_model_stats[model]['metrics'][metric] for model in model_names]
        
        df = pd.DataFrame(data, index=model_names)
        
        # 绘制柱状图
        ax = df.plot(kind='bar', figsize=(12, 6))
        plt.title('模型性能指标对比')
        plt.xlabel('模型名称')
        plt.ylabel('指标值')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=0)
        plt.legend(title='评价指标')
        plt.grid(axis='y', alpha=0.3)
        
        # 在柱子上添加数值标签
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', padding=3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(CHART_OUTPUT_DIR, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"性能指标对比图已保存: {os.path.join(CHART_OUTPUT_DIR, 'performance_metrics.png')}")
    
    def plot_inference_times(self, overall_model_stats):
        """绘制推理时间对比图"""
        model_names = list(overall_model_stats.keys())
        inference_times = [overall_model_stats[model]['inference_stats']['avg_inference_time'] * 1000 for model in model_names]
        
        # 绘制柱状图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, inference_times, color='skyblue')
        plt.title('模型平均推理时间对比')
        plt.xlabel('模型名称')
        plt.ylabel('平均推理时间 (ms)')
        plt.grid(axis='y', alpha=0.3)
        
        # 在柱子上添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(CHART_OUTPUT_DIR, 'inference_times.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"推理时间对比图已保存: {os.path.join(CHART_OUTPUT_DIR, 'inference_times.png')}")

def save_results_to_file(validator, all_results, output_path):
    """将验证结果保存到文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("模型验证结果报告\n")
            f.write("="*60 + "\n\n")
            
            # 写入模型信息
            f.write("模型信息:\n")
            f.write(f"  分类模型数量: {len(validator.classification_models)}\n")
            for i, model_info in enumerate(validator.classification_models):
                f.write(f"  分类模型{i+1}: {model_info['path']}\n")
            f.write("\n")
            
            # 写入验证配置信息
            f.write("验证配置:\n")
            f.write(f"  使用TTA: {'是' if USE_TTA else '否'}\n")
            f.write(f"  当前阈值: {OVERFLOW_THRESHOLD:.2f}\n")
            f.write(f"  阈值调优: {'是' if TUNE_THRESHOLD else '否'}\n")
            if TUNE_THRESHOLD:
                f.write(f"  阈值搜索范围: {THRESHOLD_RANGE[0]:.2f} - {THRESHOLD_RANGE[1]:.2f}\n")
                f.write(f"  阈值搜索步长: {THRESHOLD_STEP:.2f}\n")
            f.write("\n")
            
            # 写入验证文件夹信息
            f.write("验证文件夹信息:\n")
            for i, folder in enumerate(VALIDATION_FOLDERS):
                f.write(f"  验证文件夹{i+1}: {folder}\n")
            f.write("\n")
            
            # 写入各文件夹的详细结果
            for results in all_results:
                if results is None:
                    continue
                
                f.write("="*60 + "\n")
                f.write(f"验证结果: {results['folder_name']}\n")
                f.write("="*60 + "\n")
                f.write(f"总图像数: {results['total_images']}\n\n")
                
                # 写入各模型的详细统计
                f.write("各模型详细统计:\n")
                for model_name, model_result in results['model_results'].items():
                    if not model_result['true_labels']:
                        continue
                    
                    metrics = validator.calculate_metrics(model_result['true_labels'], model_result['predicted_labels'])
                    avg_inference_time = np.mean(model_result['inference_times'])
                    
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
                    f.write(f"  精确率: {metrics['precision']:.4f}\n")
                    f.write(f"  召回率: {metrics['recall']:.4f}\n")
                    f.write(f"  F1分数: {metrics['f1']:.4f}\n")
                    f.write(f"  平均推理时间: {avg_inference_time*1000:.2f}ms\n")
                    
                    # 写入混淆矩阵
                    f.write("  混淆矩阵:\n")
                    cm = metrics['confusion_matrix']
                    f.write(f"    [[{cm[0][0]}  {cm[0][1]}]\n")
                    f.write(f"     [{cm[1][0]}  {cm[1][1]}]]\n")
            
            # 写入总体摘要
            f.write("\n" + "="*60 + "\n")
            f.write("总体摘要\n")
            f.write("="*60 + "\n")
            
            # 合并所有结果
            overall_results = {
                'total_images': 0,
                'model_results': {model['name']: {'true_labels': [], 'predicted_labels': [], 'inference_times': [], 'overflow_probs': []} for model in validator.classification_models}
            }
            
            for results in all_results:
                if results is None:
                    continue
                
                overall_results['total_images'] += results['total_images']
                
                for model_name, model_result in results['model_results'].items():
                    overall_results['model_results'][model_name]['true_labels'].extend(model_result['true_labels'])
                    overall_results['model_results'][model_name]['predicted_labels'].extend(model_result['predicted_labels'])
                    overall_results['model_results'][model_name]['inference_times'].extend(model_result['inference_times'])
                    overall_results['model_results'][model_name]['overflow_probs'].extend(model_result['overflow_probs'])
            
            f.write(f"总图像数: {overall_results['total_images']}\n")
            f.write(f"使用TTA: {'是' if USE_TTA else '否'}\n")
            f.write(f"当前阈值: {OVERFLOW_THRESHOLD:.2f}\n\n")
            
            # 计算各模型的总体统计
            overall_model_stats = validator.generate_model_statistics(overall_results)
            
            # 写入各模型的总体统计
            f.write("各模型总体统计:\n")
            for model_name, stats in overall_model_stats.items():
                metrics = stats['metrics']
                inference_stats = stats['inference_stats']
                
                f.write(f"\n{model_name}:\n")
                f.write(f"  准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
                f.write(f"  精确率: {metrics['precision']:.4f}\n")
                f.write(f"  召回率: {metrics['recall']:.4f}\n")
                f.write(f"  F1分数: {metrics['f1']:.4f}\n")
                f.write(f"  平均推理时间: {inference_stats['avg_inference_time']*1000:.2f}ms\n")
                f.write(f"  总推理时间: {inference_stats['total_inference_time']:.2f}s\n")
                
                # 写入最优阈值信息
                if TUNE_THRESHOLD and stats['optimal_threshold'] is not None:
                    f.write(f"  最优阈值: {stats['optimal_threshold']:.2f}\n")
                    f.write(f"  最优F1分数: {stats['optimal_metrics']['f1']:.4f}\n")
                    f.write(f"  最优精确率: {stats['optimal_metrics']['precision']:.4f}\n")
                    f.write(f"  最优召回率: {stats['optimal_metrics']['recall']:.4f}\n")
            
            # 写入模型对比表格
            f.write("\n" + "="*60 + "\n")
            f.write("模型性能对比\n")
            f.write("="*60 + "\n")
            
            # 写入对比表格
            f.write(f"{'模型名称':<10} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'平均推理时间(ms)':<20}\n")
            f.write(f"-{'-'*10}-{'-'*10}-{'-'*10}-{'-'*10}-{'-'*10}-{'-'*20}\n")
            
            for model_name, stats in overall_model_stats.items():
                metrics = stats['metrics']
                inference_stats = stats['inference_stats']
                f.write(f"{model_name:<10} {metrics['accuracy']*100:<9.2f}% {metrics['precision']:<9.4f} {metrics['recall']:<9.4f} {metrics['f1']:<9.4f} {inference_stats['avg_inference_time']*1000:<19.2f}\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"\n结果报告已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"\n保存结果报告失败: {str(e)}")
        return False

def main():
    print("="*60)
    print("模型验证脚本")
    print("="*60)
    print()
    
    # 创建验证器
    try:
        validator = ModelValidator(CLASSIFICATION_MODEL_PATHS)
    except Exception as e:
        print(f"模型加载失败，程序终止: {str(e)}")
        return
    
    # 验证所有文件夹
    all_results = []
    
    for i, folder_path in enumerate(VALIDATION_FOLDERS):
        if os.path.exists(folder_path):
            results = validator.validate_folder(folder_path, f"验证文件夹{i+1}")
            validator.print_summary(results)
            all_results.append(results)
        else:
            print(f"验证文件夹不存在: {folder_path}")
            all_results.append(None)
    
    # 打印总体摘要
    if any(all_results):
        validator.print_overall_summary(all_results)
    
    # 保存结果到文件
    save_results_to_file(validator, all_results, RESULT_OUTPUT_PATH)
    
    print("验证完成！")

if __name__ == "__main__":
    main()