import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
import time


class PotDetector:
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """
        初始化锅具检测器

        参数:
            model_path: YOLO模型路径，如果为None则尝试加载默认模型
            confidence_threshold: 置信度阈值
        """
        self.confidence_threshold = confidence_threshold
        self.model_loaded = False
        self.model = None
        self.pot_class_id = None  # 锅具的类别ID

        # 如果提供了模型路径，则加载模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # 尝试加载默认模型
            self.load_default_model()

    def load_default_model(self):
        """尝试加载默认检测模型（如果存在）"""
        # 检查默认模型文件是否存在
        default_model_path = 'pot_424.pt'  # 锅具检测模型

        if os.path.exists(default_model_path):
            try:
                self.model = YOLO(default_model_path)
                self.model_loaded = True
                print("默认锅具检测模型已加载")

                # 尝试确定锅具的类别ID
                self.determine_pot_class_id()
            except Exception as e:
                print(f"无法加载默认检测模型: {str(e)}")
                self.model_loaded = False
        else:
            print("默认模型文件不存在，请提供模型路径")
            self.model_loaded = False

    def load_model(self, model_path):
        """加载指定路径的YOLO模型"""
        try:
            self.model = YOLO(model_path)
            self.model_loaded = True
            print(f"模型已加载: {model_path}")

            # 尝试确定锅具的类别ID
            self.determine_pot_class_id()
        except Exception as e:
            print(f"无法加载模型 {model_path}: {str(e)}")
            self.model_loaded = False

    def determine_pot_class_id(self):
        """确定锅具在模型中的类别ID"""
        # 尝试常见的锅具类别名称
        pot_keywords = ['pot', 'pan', '锅', 'pot_424']

        for i, name in self.model.names.items():
            if any(keyword in name.lower() for keyword in pot_keywords):
                self.pot_class_id = i
                print(f"检测到锅具类别: {name} (ID: {i})")
                return

        # 如果没有找到明确的锅具类别，使用第一个类别
        if len(self.model.names) > 0:
            self.pot_class_id = 0
            print(f"未明确检测到锅具类别，使用第一个类别: {self.model.names[0]} (ID: 0)")
        else:
            print("警告: 模型没有定义任何类别")

    def expand_bbox(self, x1, y1, x2, y2, img_width, img_height, expand_ratio=1.35):
        """
        扩展边界框

        参数:
            x1, y1, x2, y2: 原始边界框坐标
            img_width, img_height: 图像尺寸
            expand_ratio: 扩展比例

        返回:
            扩展后的边界框坐标
        """
        # 计算原始边界框的中心和尺寸
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 计算扩展后的尺寸
        new_width = width * expand_ratio
        new_height = height * expand_ratio

        # 计算扩展后的边界框坐标
        new_x1 = max(0, center_x - new_width / 2)
        new_y1 = max(0, center_y - new_height / 2)
        new_x2 = min(img_width, center_x + new_width / 2)
        new_y2 = min(img_height, center_y + new_height / 2)

        return int(new_x1), int(new_y1), int(new_x2), int(new_y2)

    def detect_and_crop_pots(self, frame):
        """
        检测锅具并裁剪

        参数:
            frame: 输入图像帧

        返回:
            cropped_images: 裁剪出的锅具图像列表（无边框）
            detections: 检测结果列表，包含位置信息
        """
        if not self.model_loaded:
            print("模型未加载，无法进行检测")
            return [], []

        # 获取图像尺寸
        img_height, img_width = frame.shape[:2]

        # 使用YOLO模型进行预测
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        # 初始化检测结果列表和裁剪图像列表
        detections = []
        cropped_images = []

        # 处理检测结果
        for result in results:
            # 获取检测到的边界框
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    # 只处理锅具类别
                    if class_id == self.pot_class_id:
                        # 扩展边界框
                        expanded_x1, expanded_y1, expanded_x2, expanded_y2 = self.expand_bbox(
                            x1, y1, x2, y2, img_width, img_height, expand_ratio=1.35
                        )

                        # 裁剪图像（不添加任何边框）
                        cropped_img = frame[expanded_y1:expanded_y2, expanded_x1:expanded_x2]

                        # 计算检测框中心坐标
                        center_x = (x1 + x2) / 2
                        
                        # 根据中心x坐标判断位置（左A、中B、右C）
                        if center_x < img_width / 3:
                            position = 'A'  # 左侧
                        elif center_x < img_width * 2 / 3:
                            position = 'B'  # 中间
                        else:
                            position = 'C'  # 右侧

                        # 记录检测结果，包含位置信息
                        detection = {
                            'class_id': class_id,
                            'class_name': self.model.names[class_id],
                            'confidence': confidence,
                            'original_box': (int(x1), int(y1), int(x2), int(y2)),
                            'expanded_box': (expanded_x1, expanded_y1, expanded_x2, expanded_y2),
                            'cropped_size': cropped_img.shape[:2] if cropped_img.size > 0 else (0, 0),
                            'position': position  # 添加位置信息
                        }
                        detections.append(detection)

                        # 添加到裁剪图像列表
                        if cropped_img.size > 0:
                            cropped_images.append(cropped_img)

        return cropped_images, detections


def process_video_for_pots(video_path, output_dir, pot_detector, save_interval=1, show_preview=True):
    """
    处理视频并保存检测到的锅具图像（无边框）

    参数:
        video_path: 视频文件路径
        output_dir: 输出目录
        pot_detector: 锅具检测器实例
        save_interval: 保存间隔（每隔多少帧保存一次）
        show_preview: 是否显示实时预览窗口
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"视频信息: {total_frames}帧, {fps:.2f}FPS, 时长: {duration:.2f}秒")

    frame_count = 0
    saved_count = 0
    start_time = time.time()

    # 处理视频帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔save_interval帧处理一次
        if frame_count % save_interval == 0:
            # 进行锅具检测和裁剪
            cropped_images, detections = pot_detector.detect_and_crop_pots(frame)

            # 保存裁剪出的锅具图像（无边框）
            for i, (cropped_img, detection) in enumerate(zip(cropped_images, detections)):
                # 获取位置信息（A/B/C）
                position = detection['position']
                
                # 生成位置子目录（直接使用位置作为二级目录）
                position_dir = os.path.join(output_dir, position)
                
                # 创建位置子目录
                if not os.path.exists(position_dir):
                    os.makedirs(position_dir, exist_ok=True)

                # 获取视频名（output_dir的最后一个目录名）
                video_name = os.path.basename(output_dir)
                
                # 生成输出文件名：视频名+帧数+位置（a,b,c）
                filename = f"{video_name}_{frame_count:06d}_{position.lower()}.png"
                output_path = os.path.join(position_dir, filename)

                # 保存图像（不添加任何边框）
                cv2.imwrite(output_path, cropped_img)
                saved_count += 1

                # 打印进度
                elapsed_time = time.time() - start_time
                fps_processed = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f"已处理 {frame_count}/{total_frames} 帧, 检测到 {len(cropped_images)} 个锅具, "
                      f"位置: {position}, 置信度: {detection['confidence']:.2f}, "
                      f"保存为 {filename}, 处理速度: {fps_processed:.2f} FPS")

            # 显示预览窗口（显示原始帧，不添加检测框）
            if show_preview:
                preview_frame = cv2.resize(frame, (960, 540))
                cv2.imshow('Pot Detection Preview', preview_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        frame_count += 1

    # 释放资源
    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print(f"处理完成! 共保存了 {saved_count} 张锅具图像, 总耗时: {total_time:.2f} 秒")


def main():
    # 模型路径 - 如果为None则尝试加载默认模型
    model_path = r"F:\work_area\___overflow\code_\mod_2_old\pot_424.pt" # 替换为您的锅具检测模型路径
 
    # 输入视频文件夹路径
    input_folder = r"F:\work_area\___overflow\code_\video_cut\a17_200_300+++"  # 替换为您的视频文件夹路径

    # 输出根目录
    output_root_dir = r"F:\work_area\___Labellmg\l2026_1_5_17tiaowen" # 替换为您的输出根目录

    # 支持的视频格式
    video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

    # 初始化锅具检测器
    pot_detector = PotDetector(model_path=model_path, confidence_threshold=0.5)

    if not pot_detector.model_loaded:
        print("无法加载模型，程序退出")
        return

    # 获取文件夹中的所有视频文件
    video_files = []
    for file in os.listdir(input_folder):
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext in video_formats:
            video_files.append(file)

    # 按文件名排序，确保按照视频顺序处理
    video_files.sort()

    print(f"找到 {len(video_files)} 个视频文件，开始处理...")

    # 逐个处理视频
    for idx, video_file in enumerate(video_files):
        # 视频完整路径
        video_path = os.path.join(input_folder, video_file)
        
        # 为每个视频创建单独的输出文件夹（简化结构：直接视频名作为一级目录）
        video_name = os.path.splitext(video_file)[0]
        output_dir = os.path.join(output_root_dir, video_name)
        
        print(f"\n--- 处理视频 {idx+1}/{len(video_files)}: {video_file} ---")
        print(f"输出目录: {output_dir}")

        # 处理视频
        process_video_for_pots(
            video_path,
            output_dir,
            pot_detector,
            save_interval=1,  # 每1帧处理一次
            show_preview=False  # 不显示实时预览，提高处理速度
        )

    print("\n所有视频处理完成！")


if __name__ == "__main__":
    main()