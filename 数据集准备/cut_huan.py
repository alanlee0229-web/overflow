import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
import time


class PotDetector:
    """高性能锅具检测器，基于YOLO模型实现实时检测"""

    def __init__(self, model_path=None, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.model_loaded = False
        self.model = None
        self.pot_class_id = None

        # 自动模型加载逻辑
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.load_default_model()

    def load_default_model(self):
        """加载默认锅具检测模型"""
        default_model_path = r'F:\work_area\___overflow\code_\mod_2_old\pot_424.pt'
        if os.path.exists(default_model_path):
            try:
                self.model = YOLO(default_model_path)
                self.model_loaded = True
                print(" 默认锅具检测模型加载成功")
                self.determine_pot_class_id()
            except Exception as e:
                print(f" 加载默认模型失败: {e}")
                self.model_loaded = False
        else:
            print(" 未找到默认模型文件，请提供有效路径")
            self.model_loaded = False

    def load_model(self, model_path):
        """加载指定路径的YOLO模型"""
        try:
            self.model = YOLO(model_path)
            self.model_loaded = True
            print(f" 模型加载成功: {model_path}")
            self.determine_pot_class_id()
        except Exception as e:
            print(f" 模型加载失败 {model_path}: {e}")
            self.model_loaded = False

    def determine_pot_class_id(self):
        """智能识别锅具类别ID"""
        pot_keywords = ['pot', 'pan', '锅', 'pot_424']
        for i, name in self.model.names.items():
            if any(keyword in name.lower() for keyword in pot_keywords):
                self.pot_class_id = i
                print(f"🔍 识别到锅具类别: {name} (ID: {i})")
                return

        if len(self.model.names) > 0:
            self.pot_class_id = 0
            print(f"未识别到明确锅具类别，使用首类别: {self.model.names[0]} (ID: 0)")
        else:
            print(" 错误: 模型中未定义任何类别")

    def expand_bbox(self, x1, y1, x2, y2, img_width, img_height, expand_ratio=1.08):
        """扩展边界框以确保完整包含锅具"""
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        new_width = width * expand_ratio
        new_height = height * expand_ratio

        new_x1 = max(0, center_x - new_width / 2)
        new_y1 = max(0, center_y - new_height / 2)
        new_x2 = min(img_width, center_x + new_width / 2)
        new_y2 = min(img_height, center_y + new_height / 2)

        return int(new_x1), int(new_y1), int(new_x2), int(new_y2)

    def detect_and_process_pots(self, frame, target_size=(224, 224), apply_mask=False, mask_processor=None):
        """
        核心检测与处理方法：检测锅具并直接返回处理后的图像
        """
        if not self.model_loaded:
            print(" 模型未加载，无法执行检测")
            return [], []

        img_height, img_width = frame.shape[:2]
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        processed_images = []
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    if class_id == self.pot_class_id:
                        # 扩展边界框
                        expanded_x1, expanded_y1, expanded_x2, expanded_y2 = self.expand_bbox(
                            x1, y1, x2, y2, img_width, img_height
                        )

                        # 裁剪图像
                        cropped_img = frame[expanded_y1:expanded_y2, expanded_x1:expanded_x2]

                        if cropped_img.size > 0:
                            # 调整尺寸至目标大小
                            resized_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)

                            # 直接应用掩膜处理
                            if apply_mask and mask_processor is not None:
                                final_img = mask_processor.apply_ring_mask(resized_img)
                            else:
                                final_img = resized_img

                            processed_images.append(final_img)

                            # 记录检测信息
                            detection = {
                                'confidence': float(confidence),
                                'original_bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'expanded_bbox': (expanded_x1, expanded_y1, expanded_x2, expanded_y2),
                                'final_size': final_img.shape[:2]
                            }
                            detections.append(detection)

        return processed_images, detections


class RingMaskProcessor:
    """环形掩膜处理器：专为锅具图像设计的预处理工具"""

    def apply_ring_mask(self, image, inner_ratio=0.36, outer_ratio=0.08):
        """
        应用环形掩膜到输入图像，突出锅具主体特征
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        radius = max(height, width) // 2

        # 计算内外环半径
        radius_inner = int(radius * inner_ratio)
        radius_outer = int(radius * outer_ratio)

        # 创建外环
        outer_circle = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(outer_circle, center, radius + radius_outer, 255, -1)

        # 创建内环
        inner_circle = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(inner_circle, center, radius - radius_inner, 255, -1)

        # 生成环形掩膜
        ring_mask = cv2.subtract(outer_circle, inner_circle)

        # 应用掩膜
        result = cv2.bitwise_and(image, image, mask=ring_mask)
        return result

    def process_image_directory(self, input_dir, output_dir):
        """
        批量处理目录中的图像文件
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        processed_count = 0

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(supported_exts):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)

                try:
                    image = cv2.imread(input_path)
                    if image is None:
                        continue

                    # 应用掩膜处理
                    processed_image = self.apply_ring_mask(image)
                    cv2.imwrite(output_path, processed_image)
                    processed_count += 1

                except Exception as e:
                    print(f"处理图像 {filename} 时出错: {e}")

        print(f" 批量处理完成，共处理 {processed_count} 张图像")


class PotDetectionPipeline:
    """锅具检测流水线：协调检测与处理流程的核心控制器"""

    def __init__(self, detector_config, processor_config=None):
        self.detector = PotDetector(**detector_config)
        self.processor = RingMaskProcessor() if processor_config else None
        self.processing_stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detected_pots': 0,
            'start_time': None,
            'end_time': None
        }

    def process_video(self, video_path, output_dir, save_interval=1, show_preview=False):
        """
        处理视频文件并直接输出掩膜处理后的锅具图像
        """
        if not self.detector.model_loaded:
            print(" 检测器模型未加载，无法处理视频")
            return False

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(" 无法打开视频文件")
            return False

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"📊 视频信息: {total_frames}帧, {fps:.2f}FPS, 时长: {duration:.2f}秒")

        # 初始化统计信息
        self.processing_stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'detected_pots': 0,
            'start_time': time.time(),
            'end_time': None
        }

        frame_count = 0
        saved_count = 0

        print("🚀 开始处理视频...")

        # 处理视频帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % save_interval == 0:
                # 检测并处理锅具图像
                processed_images, detections = self.detector.detect_and_process_pots(
                    frame,
                    target_size=(224, 224),
                    apply_mask=True,
                    mask_processor=self.processor
                )

                # 保存处理后的图像
                for i, processed_img in enumerate(processed_images):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"masked_pot_{timestamp}_f{frame_count:06d}_{i}.png"
                    output_path = os.path.join(output_dir, filename)

                    cv2.imwrite(output_path, processed_img)
                    saved_count += 1
                    self.processing_stats['detected_pots'] += 1

                self.processing_stats['processed_frames'] += 1

                # 显示处理进度
                if frame_count % 30 == 0:
                    self._display_progress(frame_count, total_frames, saved_count)

                # 实时预览
                if show_preview:
                    preview_frame = cv2.resize(frame, (960, 540))
                    cv2.imshow('锅具检测预览', preview_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            frame_count += 1

        # 完成处理
        self.processing_stats['end_time'] = time.time()
        self._display_final_stats()

        # 释放资源
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()

        return True

    def _display_progress(self, current_frame, total_frames, saved_count):
        """显示处理进度信息"""
        elapsed_time = time.time() - self.processing_stats['start_time']
        frames_per_second = current_frame / elapsed_time if elapsed_time > 0 else 0
        progress_percent = (current_frame / total_frames) * 100

        print(f"📈 进度: {current_frame}/{total_frames} ({progress_percent:.1f}%) | "
              f"速度: {frames_per_second:.2f} FPS | "
              f"检测到: {saved_count} 个锅具")

    def _display_final_stats(self):
        """显示最终统计信息"""
        total_time = self.processing_stats['end_time'] - self.processing_stats['start_time']
        avg_fps = self.processing_stats['processed_frames'] / total_time if total_time > 0 else 0

        print("\n" + "=" * 50)
        print(" 处理完成!")
        print("=" * 50)
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均处理速度: {avg_fps:.2f} FPS")
        print(f"处理帧数: {self.processing_stats['processed_frames']}")
        print(f"检测到锅具总数: {self.processing_stats['detected_pots']}")
        # print(f"输出目录: {os.path.abspath(output_dir)}")
        print("=" * 50)

    def batch_process_videos(self, video_directory, output_base_dir):
        """批量处理多个视频文件"""
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')

        if not os.path.exists(output_base_dir):
            os.makedirs(output_base_dir)

        processed_count = 0
        for filename in os.listdir(video_directory):
            if filename.lower().endswith(video_extensions):
                video_path = os.path.join(video_directory, filename)
                video_name = os.path.splitext(filename)[0]
                output_dir = os.path.join(output_base_dir, f"output_{video_name}")

                print(f"\n🎬 开始处理视频: {filename}")
                success = self.process_video(video_path, output_dir, show_preview=False)

                if success:
                    processed_count += 1
                    print(f" 完成处理: {filename}")
                else:
                    print(f" 处理失败: {filename}")

        print(f"\n🎉 批量处理完成，成功处理 {processed_count} 个视频文件")


class ConfigManager:
    """配置管理器：统一管理所有运行参数"""

    DEFAULT_CONFIG = {
        'detector': {
            'model_path': 'pot_424.pt',
            'confidence_threshold': 0.5
        },
        'processor': {
            'inner_ratio': 0.36,
            'outer_ratio': 0.08
        },
        'pipeline': {
            'save_interval': 1,
            'show_preview': False,
            'output_format': 'png'
        }
    }

    def __init__(self, config_file=None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)

    def load_config(self, config_file):
        """从JSON文件加载配置"""
        try:
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                self._deep_update(self.config, user_config)
            print(f" 配置加载成功: {config_file}")
        except Exception as e:
            print(f"  配置加载失败，使用默认配置: {e}")

    def save_config(self, config_file):
        """保存配置到JSON文件"""
        try:
            import json
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f" 配置保存成功: {config_file}")
        except Exception as e:
            print(f" 配置保存失败: {e}")

    def _deep_update(self, base, update):
        """深度更新字典"""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def get_detector_config(self):
        return self.config['detector']

    def get_processor_config(self):
        return self.config['processor']

    def get_pipeline_config(self):
        return self.config['pipeline']


def main():
    """
    主应用程序：锅具检测与处理系统的入口点
    """
    print("=" * 60)
    print("🔧 锅具检测与掩膜处理系统")
    print("=" * 60)

    # 初始化配置管理器
    config_manager = ConfigManager('config.json')

    # 获取配置参数
    detector_config = config_manager.get_detector_config()
    processor_config = config_manager.get_processor_config()
    pipeline_config = config_manager.get_pipeline_config()

    # 创建处理流水线
    pipeline = PotDetectionPipeline(detector_config, processor_config)

    # 设置输入输出路径
    video_path = input("请输入视频文件路径: ").strip().strip('"')
    output_dir = input("请输入输出目录路径: ").strip().strip('"')

    if not video_path or not output_dir:
        print(" 错误: 路径不能为空")
        return

    # 验证路径有效性
    if not os.path.exists(video_path):
        print(f" 错误: 视频文件不存在 {video_path}")
        return

    # 执行处理
    print("\n🚀 开始处理...")
    success = pipeline.process_video(
        video_path=video_path,
        output_dir=output_dir,
        save_interval=pipeline_config['save_interval'],
        show_preview=pipeline_config['show_preview']
    )

    if success:
        print("\n🎉 处理完成!")
        print(f"📁 输出目录: {os.path.abspath(output_dir)}")
    else:
        print("\n 处理失败，请检查错误信息")


def batch_process_mode():
    """批量处理模式"""
    print("=" * 60)
    print("🔧 批量处理模式")
    print("=" * 60)

    config_manager = ConfigManager()
    detector_config = config_manager.get_detector_config()
    processor_config = config_manager.get_processor_config()

    pipeline = PotDetectionPipeline(detector_config, processor_config)

    input_dir = input("请输入视频目录路径: ").strip().strip('"')
    output_dir = input("请输入输出根目录: ").strip().strip('"')

    if not os.path.exists(input_dir):
        print(f" 错误: 输入目录不存在 {input_dir}")
        return

    pipeline.batch_process_videos(input_dir, output_dir)


if __name__ == "__main__":
    # 选择运行模式
    print("请选择运行模式:")
    print("1. 单个视频处理")
    print("2. 批量视频处理")

    choice = input("请输入选择 (1/2): ").strip()

    if choice == "2":
        batch_process_mode()
    else:
        main()