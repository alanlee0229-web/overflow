"""
YOLO 数据集图像推理验证脚本
对数据集图像进行推理，与标签进行比对验证
"""

import cv2
import os
import glob
import csv
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path


# ==================== 配置参数（修改这里） ====================

# 模型路径
MODEL_PATH = "Train_obstacle/0103--Train_obstacle--w12.0-20.0_h28.0-40.0/weights/best.pt"

# 数据集路径（YOLO格式：images和labels文件夹同级）
DATASET_PATH = "/root/autodl-tmp/dataset"

# 图像子文件夹名（相对于DATASET_PATH）
IMAGES_FOLDER = "images"

# 标签子文件夹名（相对于DATASET_PATH）
LABELS_FOLDER = "labels"

# 输出文件夹路径（保存可视化结果和统计报告）
OUTPUT_FOLDER = "/root/autodl-tmp/validate_results"

# 是否保存可视化图像（检测框+标签框对比）
SAVE_IMAGES = True

# 置信度阈值
CONF_THRESHOLD = 0.5

# NMS IOU阈值
IOU_THRESHOLD = 0.45

# 推理图像尺寸
IMG_SIZE = 640

# 设备 ("0", "1", "cpu")
DEVICE = "0"

# ==================== 评估参数 ====================

# 匹配IOU阈值（预测框与标签框IOU大于此值视为匹配成功）
MATCH_IOU_THRESHOLD = 0.5

# ==================== 可视化参数 ====================

# 标签框颜色 (BGR格式) - 绿色
GT_COLOR = (0, 255, 0)

# 预测框颜色 (BGR格式) - 红色  
PRED_COLOR = (0, 0, 255)

# 匹配成功的预测框颜色 - 蓝色
MATCHED_COLOR = (255, 0, 0)

# 边界框线宽
LINE_WIDTH = 2

# ==============================================================


def calculate_iou(box1, box2):
    """
    计算两个框的IOU
    box格式: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area


def parse_yolo_label(label_path, img_width, img_height):
    """
    解析YOLO格式标签文件
    返回: [(class_id, x1, y1, x2, y2), ...]
    """
    labels = []
    
    if not os.path.exists(label_path):
        return labels
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                labels.append((class_id, x1, y1, x2, y2))
    
    return labels


def match_predictions_to_labels(predictions, labels, iou_threshold):
    """
    将预测框与标签框进行匹配
    返回: matched_preds, unmatched_preds, unmatched_labels
    """
    if len(predictions) == 0:
        return [], [], labels.copy()
    
    if len(labels) == 0:
        return [], predictions.copy(), []
    
    # 计算IOU矩阵
    iou_matrix = np.zeros((len(predictions), len(labels)))
    for i, pred in enumerate(predictions):
        for j, label in enumerate(labels):
            iou_matrix[i, j] = calculate_iou(pred[1:5], label[1:5])
    
    matched_preds = []
    matched_labels_idx = set()
    unmatched_preds = []
    
    # 贪婪匹配：按IOU从大到小匹配
    while True:
        if iou_matrix.size == 0:
            break
        
        max_iou = iou_matrix.max()
        if max_iou < iou_threshold:
            break
        
        pred_idx, label_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        
        matched_preds.append((predictions[pred_idx], labels[label_idx], max_iou))
        matched_labels_idx.add(label_idx)
        
        # 将已匹配的行列设为0
        iou_matrix[pred_idx, :] = 0
        iou_matrix[:, label_idx] = 0
    
    # 找出未匹配的预测框
    for i, pred in enumerate(predictions):
        matched = False
        for mp in matched_preds:
            if pred == mp[0]:
                matched = True
                break
        if not matched:
            unmatched_preds.append(pred)
    
    # 找出未匹配的标签框
    unmatched_labels = [labels[i] for i in range(len(labels)) if i not in matched_labels_idx]
    
    return matched_preds, unmatched_preds, unmatched_labels


def inference_single_image(model, image_path, label_path, output_path=None):
    """对单张图像进行推理并与标签比对"""
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_height, img_width = img.shape[:2]
    
    # 解析标签
    labels = parse_yolo_label(label_path, img_width, img_height)
    
    # 推理
    results = model.predict(
        source=img,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False
    )
    
    result = results[0]
    boxes = result.boxes
    
    # 解析预测结果
    predictions = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            predictions.append((cls, x1, y1, x2, y2, conf))
    
    # 匹配预测框与标签框
    matched, unmatched_preds, unmatched_labels = match_predictions_to_labels(
        predictions, labels, MATCH_IOU_THRESHOLD
    )
    
    # 计算指标
    tp = len(matched)  # True Positive
    fp = len(unmatched_preds)  # False Positive
    fn = len(unmatched_labels)  # False Negative
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 保存可视化图像
    if output_path:
        vis_img = img.copy()
        
        # 绘制标签框（绿色）
        for label in labels:
            x1, y1, x2, y2 = int(label[1]), int(label[2]), int(label[3]), int(label[4])
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), GT_COLOR, LINE_WIDTH)
            cv2.putText(vis_img, f"GT:{label[0]}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 1)
        
        # 绘制匹配成功的预测框（蓝色）
        for pred, _, iou in matched:
            x1, y1, x2, y2 = int(pred[1]), int(pred[2]), int(pred[3]), int(pred[4])
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), MATCHED_COLOR, LINE_WIDTH)
            cv2.putText(vis_img, f"P:{pred[5]:.2f} IoU:{iou:.2f}", (x1, y2+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, MATCHED_COLOR, 1)
        
        # 绘制未匹配的预测框（红色 - 误检）
        for pred in unmatched_preds:
            x1, y1, x2, y2 = int(pred[1]), int(pred[2]), int(pred[3]), int(pred[4])
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), PRED_COLOR, LINE_WIDTH)
            cv2.putText(vis_img, f"FP:{pred[5]:.2f}", (x1, y2+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRED_COLOR, 1)
        
        # 添加图例和统计信息
        legend_y = 30
        cv2.putText(vis_img, f"Green=GT  Blue=Matched  Red=FP", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_img, f"TP:{tp} FP:{fp} FN:{fn} P:{precision:.2f} R:{recall:.2f}", 
                   (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, vis_img)
    
    return {
        'num_labels': len(labels),
        'num_predictions': len(predictions),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'img_width': img_width,
        'img_height': img_height
    }


def validate_dataset():
    """验证整个数据集"""
    
    images_path = os.path.join(DATASET_PATH, IMAGES_FOLDER)
    labels_path = os.path.join(DATASET_PATH, LABELS_FOLDER)
    
    # 检查路径
    if not os.path.exists(images_path):
        print(f"错误: 图像文件夹不存在 - {images_path}")
        return
    
    if not os.path.exists(labels_path):
        print(f"错误: 标签文件夹不存在 - {labels_path}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在 - {MODEL_PATH}")
        return
    
    # 创建输出文件夹
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    if SAVE_IMAGES:
        vis_folder = os.path.join(OUTPUT_FOLDER, "visualizations")
        os.makedirs(vis_folder, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_path, ext)))
    
    if not image_files:
        print(f"错误: 文件夹内没有找到图像文件 - {images_path}")
        return
    
    image_files.sort()
    
    print("=" * 70)
    print("YOLO 数据集验证")
    print("=" * 70)
    print(f"模型: {MODEL_PATH}")
    print(f"数据集路径: {DATASET_PATH}")
    print(f"图像文件夹: {images_path}")
    print(f"标签文件夹: {labels_path}")
    print(f"找到图像数量: {len(image_files)}")
    print(f"置信度阈值: {CONF_THRESHOLD}")
    print(f"匹配IOU阈值: {MATCH_IOU_THRESHOLD}")
    print("=" * 70)
    
    # 加载模型
    print("正在加载模型...")
    model = YOLO(MODEL_PATH)
    
    # 存储所有结果
    all_results = []
    
    # 累计统计
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_labels = 0
    total_predictions = 0
    
    # 处理每张图像
    for i, image_path in enumerate(image_files):
        image_name = os.path.basename(image_path)
        
        # 构建标签路径
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(labels_path, label_name)
        
        # 输出路径
        output_path = None
        if SAVE_IMAGES:
            output_name = os.path.splitext(image_name)[0] + "_compare.jpg"
            output_path = os.path.join(vis_folder, output_name)
        
        # 推理
        stats = inference_single_image(model, image_path, label_path, output_path)
        
        if stats is None:
            print(f"[{i+1}/{len(image_files)}] 跳过: {image_name} (无法读取)")
            continue
        
        # 保存结果
        result = {
            'image_name': image_name,
            'has_label': os.path.exists(label_path),
            **stats
        }
        all_results.append(result)
        
        # 累加统计
        total_tp += stats['tp']
        total_fp += stats['fp']
        total_fn += stats['fn']
        total_labels += stats['num_labels']
        total_predictions += stats['num_predictions']
        
        # 打印进度（每10张或最后一张）
        if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
            print(f"[{i+1}/{len(image_files)}] 已处理 - 当前图像: {image_name} | "
                  f"TP:{stats['tp']} FP:{stats['fp']} FN:{stats['fn']}")
    
    # 计算总体指标
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    # 保存CSV报告
    csv_path = os.path.join(OUTPUT_FOLDER, f"验证报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['图像名称', '图像宽度', '图像高度', '标签框数', '预测框数', 
                        'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1'])
        
        for r in all_results:
            writer.writerow([
                r['image_name'],
                r['img_width'],
                r['img_height'],
                r['num_labels'],
                r['num_predictions'],
                r['tp'],
                r['fp'],
                r['fn'],
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1']:.4f}"
            ])
        
        # 写入汇总行
        writer.writerow([])
        writer.writerow(['【汇总】', '', '', total_labels, total_predictions,
                        total_tp, total_fp, total_fn,
                        f"{overall_precision:.4f}", f"{overall_recall:.4f}", f"{overall_f1:.4f}"])
    
    # 打印汇总结果
    print("\n")
    print("=" * 70)
    print("                         验证结果汇总")
    print("=" * 70)
    print(f"处理图像数量:              {len(all_results)}")
    print(f"标签框总数:                {total_labels}")
    print(f"预测框总数:                {total_predictions}")
    print("-" * 70)
    print(f"True Positive (TP):        {total_tp}")
    print(f"False Positive (FP):       {total_fp} (误检)")
    print(f"False Negative (FN):       {total_fn} (漏检)")
    print("-" * 70)
    print(f"Precision (精确率):        {overall_precision:.4f}")
    print(f"Recall (召回率):           {overall_recall:.4f}")
    print(f"F1 Score:                  {overall_f1:.4f}")
    print("=" * 70)
    print(f"验证报告已保存: {csv_path}")
    if SAVE_IMAGES:
        print(f"可视化图像已保存到: {vis_folder}")
    print("=" * 70)
    
    return all_results


if __name__ == '__main__':
    validate_dataset()
