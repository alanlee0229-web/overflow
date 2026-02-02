import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from efficientnet_pytorch import EfficientNet
import time

# 类别名称映射
class_names = {
    0: "normal",
    1: "boiling over"
}


class VideoPotDetector:
    def __init__(self, yolo_model_path, efficientnet_model_path):
        """
        初始化视频锅具检测器

        Args:
            yolo_model_path: YOLO模型路径
            efficientnet_model_path: EfficientNet模型路径
        """
        self.yolo_model = None
        self.efficientnet_model = None
        self.model_loaded = False

        # 加载模型
        self.load_models(yolo_model_path, efficientnet_model_path)

    def load_models(self, yolo_model_path, efficientnet_model_path):
        """加载YOLO和EfficientNet模型"""
        try:
            print("正在加载YOLO模型...")
            self.yolo_model = YOLO(yolo_model_path)

            print("正在加载EfficientNet模型...")
            # 创建EfficientNet模型
            self.efficientnet_model = EfficientNet.from_name('efficientnet-b0', num_classes=2)

            # 加载训练好的权重
            checkpoint = torch.load(efficientnet_model_path, map_location=torch.device('cpu'))

            # 检查状态字典格式
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # 确保模型和状态字典的数据类型一致
            self.efficientnet_model = self.efficientnet_model.float()
            state_dict = {k: v.float() for k, v in state_dict.items()}

            self.efficientnet_model.load_state_dict(state_dict)
            self.efficientnet_model.eval()

            self.model_loaded = True
            print("所有模型加载完成!")

        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model_loaded = False

    def preprocess_image(self, image):
        """预处理图像用于EfficientNet模型"""
        try:
            # 确保图像是numpy数组
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()

            # 转换为RGB（如果图像是BGR格式）
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 调整图像大小到224x224（如果还不是这个大小）
            if image.shape[0] != 224 or image.shape[1] != 224:
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

            # 归一化
            image = image.astype(np.float32) / 255.0

            # 标准化 (使用ImageNet的均值和标准差)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image = (image - mean) / std

            # 转换为tensor并调整维度 [H, W, C] -> [C, H, W]
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

            # 添加batch维度 [C, H, W] -> [1, C, H, W]
            image_tensor = image_tensor.unsqueeze(0)

            return image_tensor

        except Exception as e:
            print(f"图像预处理失败: {e}")
            return None

    def classify_region(self, region_image):
        """对单个区域进行分类"""
        if not self.model_loaded or region_image.size == 0:
            return None

        try:
            # 预处理图像
            blob = cv2.dnn.blobFromImage(region_image, scalefactor=1 / 255, size=(224, 224), swapRB=True)
            input_tensor = torch.from_numpy(blob)

            if input_tensor is None:
                return None

            # 确保输入张量与模型的数据类型一致
            if self.efficientnet_model is not None:
                input_tensor = input_tensor.to(next(self.efficientnet_model.parameters()).dtype)

            # 推理
            with torch.no_grad():
                outputs = self.efficientnet_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                max_prob, predicted_idx = torch.max(probabilities, dim=1)

                predicted_class_idx = predicted_idx.item()
                confidence = max_prob.item()
                predicted_class = class_names.get(predicted_class_idx, "unknown")

                return {
                    'class_index': predicted_class_idx,
                    'class_name': predicted_class,
                    'confidence': confidence
                }

        except Exception as e:
            print(f"分类失败: {e}")
            return None

    def process_frame(self, frame):
        """处理单帧图像"""
        if not self.model_loaded or frame is None:
            return frame

        # 使用YOLO检测锅具
        results = self.yolo_model(frame)

        # 处理每个检测结果
        for i, result in enumerate(results):
            boxes = result.boxes.cpu().numpy()

            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 扩大检测框
                scale_factor = 1.08
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                width_patch = x2 - x1
                height_patch = y2 - y1

                x1_new = int(center_x - width_patch * scale_factor / 2)
                y1_new = int(center_y - height_patch * scale_factor / 2)
                x2_new = int(center_x + width_patch * scale_factor / 2)
                y2_new = int(center_y + height_patch * scale_factor / 2)

                # 确保坐标在图像范围内
                x1_new = max(0, x1_new)
                y1_new = max(0, y1_new)
                x2_new = min(frame.shape[1] - 1, x2_new)
                y2_new = min(frame.shape[0] - 1, y2_new)

                # 提取锅具图像并调整大小为224×224
                pot_image = frame[y1_new:y2_new, x1_new:x2_new]
                if pot_image.size == 0:
                    continue

                pot_resized = cv2.resize(pot_image, (224, 224), interpolation=cv2.INTER_LINEAR)

                # 获取图像的高度和宽度
                height, width = pot_resized.shape[:2]

                # 定义四个区域
                regions = {
                    'top_left': pot_resized[0:128, 0:128],
                    'top_right': pot_resized[0:128, width - 128:width],
                    'bottom_left': pot_resized[height - 128:height, 0:128],
                    'bottom_right': pot_resized[height - 128:height, width - 128:width]
                }

                # 对每个区域进行分类
                region_results = {}
                for region_name, region_image in regions.items():
                    result = self.classify_region(region_image)
                    if result:
                        region_results[region_name] = result

                # 在原图上绘制检测框
                cv2.rectangle(frame, (x1_new, y1_new), (x2_new, y2_new), (0, 255, 0), 2)

                # 计算总体结果（多数投票）
                boiling_count = sum(1 for r in region_results.values() if r['class_name'] == 'boiling over')
                normal_count = sum(1 for r in region_results.values() if r['class_name'] == 'normal')

                if boiling_count > normal_count:
                    overall_class = "Boiling Over"
                    overall_color = (0, 0, 255)  # 红色
                else:
                    overall_class = "Normal"
                    overall_color = (0, 255, 0)  # 绿色

                # 在原图上显示总体分类结果
                cv2.putText(frame, overall_class, (x1_new, y1_new - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, overall_color, 2)

                # 显示统计信息
                stats_text = f"Boiling: {boiling_count}/4, Normal: {normal_count}/4"
                cv2.putText(frame, stats_text, (x1_new, y2_new + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, overall_color, 1)

                # 在锅具框旁边显示四个区域的分类结果
                y_offset = y1_new - 40
                for region_name, result in region_results.items():
                    text = f"{region_name}: {result['class_name']}({result['confidence']:.2f})"
                    cv2.putText(frame, text, (x1_new, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset -= 15

                # 只处理第一个检测到的锅具
                break

        return frame

    def process_video(self, video_path, output_dir="output", display_scale=0.7):
        """
        处理视频文件

        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            display_scale: 显示缩放比例
        """
        if not self.model_loaded:
            print("模型未加载!")
            return

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"视频信息: {width}x{height}, {fps} FPS, 总帧数: {total_frames}")

        # 创建视频写入器
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_processed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()

        # 用于记录每帧处理时间的列表
        frame_times = []

        # 创建显示窗口
        cv2.namedWindow('Pot Detection - Video', cv2.WINDOW_NORMAL)
        
        #倍速处理287~292行
        frame_skip = 1  # 2倍速处理（每2帧处理一次），3为3倍速

        while True:
            # 记录帧开始处理的时间
            frame_start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                # 处理当前帧
                processed_frame = self.process_frame(frame)
            else:
                # 跳过处理，直接用原始帧
                processed_frame = frame

            # 写入输出视频
            out.write(processed_frame)

            # 计算帧处理时间
            frame_time = time.time() - frame_start_time
            frame_times.append(frame_time)

            # 调整显示大小
            display_width = int(width * display_scale)
            display_height = int(height * display_scale)
            display_frame = cv2.resize(processed_frame, (display_width, display_height))

            # 在显示帧上添加处理时间信息
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            avg_frame_time = sum(frame_times) / len(frame_times)
            avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

            # 在图像上显示处理时间信息
            info_text1 = f"Frame: {frame_count}/{total_frames}"
            info_text2 = f"Cur FPS: {current_fps:.2f}, Avg FPS: {avg_fps:.2f}"
            info_text3 = f"Cur Time: {frame_time * 1000:.1f}ms, Avg Time: {avg_frame_time * 1000:.1f}ms"

            cv2.putText(display_frame, info_text1, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display_frame, info_text2, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display_frame, info_text3, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 显示处理后的帧
            cv2.imshow('Pot Detection - Video', display_frame)

            # 计算并显示处理速度
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                overall_fps = frame_count / elapsed_time
                print(f"已处理 {frame_count}/{total_frames} 帧, "
                      f"当前FPS: {current_fps:.2f}, 平均FPS: {avg_fps:.2f}, "
                      f"总FPS: {overall_fps:.2f}, "
                      f"平均每帧时间: {avg_frame_time * 1000:.1f}ms")

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        total_time = time.time() - start_time

        # 计算详细的时间统计
        if frame_times:
            min_frame_time = min(frame_times) * 1000
            max_frame_time = max(frame_times) * 1000
            avg_frame_time = sum(frame_times) / len(frame_times) * 1000
            std_frame_time = np.std(frame_times) * 1000

            print("\n" + "=" * 50)
            print("处理完成!")
            print(f"总帧数: {frame_count}")
            print(f"总时间: {total_time:.2f}秒")
            print(f"平均FPS: {frame_count / total_time:.2f}")
            print(f"每帧时间统计:")
            print(f"  最短: {min_frame_time:.1f}ms")
            print(f"  最长: {max_frame_time:.1f}ms")
            print(f"  平均: {avg_frame_time:.1f}ms")
            print(f"  标准差: {std_frame_time:.1f}ms")
            print(f"输出视频已保存至: {output_path}")
            print("=" * 50)

        # 保存时间统计到文件
        stats_file = os.path.join(output_dir, f"{base_name}_time_stats.txt")
        with open(stats_file, 'w') as f:
            f.write("视频处理时间统计\n")
            f.write("=" * 30 + "\n")
            f.write(f"视频文件: {video_path}\n")
            f.write(f"总帧数: {frame_count}\n")
            f.write(f"总时间: {total_time:.2f}秒\n")
            f.write(f"平均FPS: {frame_count / total_time:.2f}\n")
            f.write(f"每帧时间统计:\n")
            f.write(f"  最短: {min_frame_time:.1f}ms\n")
            f.write(f"  最长: {max_frame_time:.1f}ms\n")
            f.write(f"  平均: {avg_frame_time:.1f}ms\n")
            f.write(f"  标准差: {std_frame_time:.1f}ms\n")

        print(f"时间统计已保存至: {stats_file}")


def main():
    # 替换为您的YOLO模型路径
    yolo_model_path = r"F:\work_area\___overflow\code_\mod_2_old\pot_424.pt" 

    # 替换为您的EfficientNet模型路径
    efficientnet_model_path = r"F:\work_area\___overflow\code_\mod_2_old\1017new\efficientnet_b0_best_lkh4.pth"  
        
    # 自动读取文件夹下所有视频文件路径
    video_folder = r"F:\work_area\___overflow\data_videos\1012\33"  # 这里填写你要读取的视频文件夹路径
    # 支持的视频格式
    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
    address = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.lower().endswith(video_exts)]

    
    # 视频文件路径
    for path in address:
        
        video_path = path # 替换为您的视频文件路径

        # 输出目录
        output_dir = r"F:\work_area\___overflow\video_output\4"

        # 创建检测器
        detector = VideoPotDetector(yolo_model_path, efficientnet_model_path)

        # 处理视频
        detector.process_video(video_path, output_dir, display_scale=0.7)


if __name__ == "__main__":
    main()