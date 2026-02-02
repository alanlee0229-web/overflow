import sys
import os
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ultralytics import YOLO
import torch
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
import matplotlib

# 配置类，用于管理所有可配置项
class Config:
    def __init__(self):
        # 模型相关配置
        self.DEFAULT_DETECTION_MODEL = 'pot_424.pt'  # 默认检测模型路径
        self.CONFIDENCE_THRESHOLD = 0.5  # 检测置信度阈值
        self.BOX_EXPAND_RATIO = 1.05  # 检测框扩展比例
        
        # 输出路径相关配置
        self.DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "saved_frames")  # 默认输出路径
        self.DEFAULT_CHART_DIR = os.path.join(os.getcwd(), "chart")  # 默认图表保存目录
        self.DEFAULT_POT_DETECTION_DIR = os.path.join(os.getcwd(), "pot_detection")  # 默认锅具检测结果保存目录
        
        # 视频相关配置
        self.SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']  # 支持的视频文件扩展名
        self.DEFAULT_FPS = 30  # 默认FPS
        
        # 推理相关配置
        self.RING_MASK_INNER_RADIUS_RATIO = 0.36  # 圆环掩膜内半径比例
        self.RING_MASK_OUTER_RADIUS_RATIO = 0.08  # 圆环掩膜外半径比例
        
        # UI相关配置
        self.BUTTON_HIGHLIGHT_DURATION = 300  # 按钮高亮持续时间(毫秒)
        self.DEFAULT_WINDOW_SIZE = (1000, 700)  # 默认窗口大小
        self.VIDEO_LABEL_MIN_SIZE = (640, 360)  # 视频标签最小尺寸

# 创建全局配置实例
config = Config()

# 锅具检测器类（从cut_to_miss_error.py整合）
class PotDetector:
    def __init__(self, model_path=None, confidence_threshold=None):
        """
        初始化锅具检测器

        参数:
            model_path: YOLO模型路径，如果为None则尝试加载默认模型
            confidence_threshold: 置信度阈值，如果为None则使用配置中的默认值
        """
        self.confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
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
        default_model_path = config.DEFAULT_DETECTION_MODEL

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
            print(f"默认模型文件不存在: {default_model_path}，请提供模型路径")
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

    def expand_bbox(self, x1, y1, x2, y2, img_width, img_height, expand_ratio=None):
        """
        扩展边界框

        参数:
            x1, y1, x2, y2: 原始边界框坐标
            img_width, img_height: 图像尺寸
            expand_ratio: 扩展比例，如果为None则使用配置中的默认值

        返回:
            扩展后的边界框坐标
        """
        # 计算原始边界框的中心和尺寸
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 计算扩展后的尺寸
        expand_ratio = expand_ratio or config.BOX_EXPAND_RATIO
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
            detections: 检测结果列表
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
                            x1, y1, x2, y2, img_width, img_height
                        )

                        # 裁剪图像（不添加任何边框）
                        cropped_img = frame[expanded_y1:expanded_y2, expanded_x1:expanded_x2]

                        # 记录检测结果
                        detection = {
                            'class_id': class_id,
                            'class_name': self.model.names[class_id],
                            'confidence': confidence,
                            'original_box': (int(x1), int(y1), int(x2), int(y2)),
                            'expanded_box': (expanded_x1, expanded_y1, expanded_x2, expanded_y2),
                            'cropped_size': cropped_img.shape[:2] if cropped_img.size > 0 else (0, 0)
                        }
                        detections.append(detection)

                        # 添加到裁剪图像列表
                        if cropped_img.size > 0:
                            cropped_images.append(cropped_img)

        return cropped_images, detections

# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 自定义进度条类，用于显示捕获点和片段
class CaptureSlider(QSlider):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.captured_segments = []  # 存储截取的片段 [(start_frame, end_frame), ...]
        self.current_segment_start = None  # 当前正在截取的片段的开始帧
        self.setTickPosition(QSlider.TicksBelow)
        self.setTickInterval(1)
        self.setStyleSheet("""QSlider {
            background-color: #333;
        }
        QSlider::groove:horizontal {
            height: 8px;
            background: #555;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            width: 12px;
            height: 12px;
            background: white;
            border: 2px solid #2196F3;
            border-radius: 6px;
            margin: -2px 0;
        }""")
    
    def paintEvent(self, event):
        # 调用父类的paintEvent来绘制基本滑块
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 获取滑块的几何信息
        rect = self.rect()
        groove_rect = self.style().subControlRect(
            QStyle.CC_Slider, self.styleOption(), QStyle.SC_SliderGroove, self)
        
        # 计算每一帧对应的像素位置
        total_frames = self.maximum() - self.minimum() + 1
        if total_frames == 0:
            return
        
        # 绘制已捕获的片段
        painter.setBrush(QColor(33, 150, 243, 150))  # 半透明蓝色
        painter.setPen(Qt.NoPen)
        
        for start_frame, end_frame in self.captured_segments:
            if start_frame is not None and end_frame is not None:
                # 计算开始和结束点在进度条上的位置
                start_x = self.frame_to_x(start_frame, groove_rect)
                end_x = self.frame_to_x(end_frame, groove_rect)
                
                # 绘制片段区域
                segment_rect = QRect(start_x, groove_rect.y(), end_x - start_x + 1, groove_rect.height())
                painter.drawRect(segment_rect)
        
        # 绘制捕获点（先绘制点，后绘制文本）
        painter.setBrush(QColor(33, 150, 243))  # 蓝色
        
        for start_frame, end_frame in self.captured_segments:
            if start_frame is not None:
                x = self.frame_to_x(start_frame, groove_rect)
                # 绘制开始点
                painter.drawEllipse(QPoint(x, groove_rect.center().y()), 5, 5)
            if end_frame is not None:
                x = self.frame_to_x(end_frame, groove_rect)
                # 绘制结束点
                painter.drawEllipse(QPoint(x, groove_rect.center().y()), 5, 5)
        
        # 绘制当前正在捕获的片段的开始点
        if self.current_segment_start is not None:
            x = self.frame_to_x(self.current_segment_start, groove_rect)
            painter.setBrush(QColor(255, 152, 0))  # 橙色
            painter.drawEllipse(QPoint(x, groove_rect.center().y()), 5, 5)
        
        # 设置绘制文本的字体和颜色（最后绘制文本，确保在最上层）
        painter.setFont(QFont("Arial", 8, QFont.Bold))  # 增大字体并加粗
        painter.setPen(QColor(255, 0, 0))  # 使用红色提高对比度
        
        # 绘制帧序号
        for start_frame, end_frame in self.captured_segments:
            if start_frame is not None:
                x = self.frame_to_x(start_frame, groove_rect)
                # 绘制开始帧序号（转换为1基索引）
                start_text = str(start_frame + 1)
                # 确保文本在可见区域内
                text_x = max(2, min(x - 10, rect.width() - 20))  # 调整x坐标确保可见
                text_y = rect.bottom() - 2  # 将文本绘制在整个组件的底部边缘
                painter.drawText(text_x, text_y, start_text)
            if end_frame is not None:
                x = self.frame_to_x(end_frame, groove_rect)
                # 绘制结束帧序号（转换为1基索引）
                end_text = str(end_frame + 1)
                # 确保文本在可见区域内
                text_x = max(2, min(x - 10, rect.width() - 20))  # 调整x坐标确保可见
                text_y = rect.bottom() - 2  # 将文本绘制在整个组件的底部边缘
                painter.drawText(text_x, text_y, end_text)
        
        # 绘制当前捕获的开始帧序号（最后绘制，确保在最上层）
        if self.current_segment_start is not None:
            x = self.frame_to_x(self.current_segment_start, groove_rect)
            # 绘制当前捕获的开始帧序号（转换为1基索引）
            start_text = str(self.current_segment_start + 1)
            # 确保文本在可见区域内
            text_x = max(2, min(x - 10, rect.width() - 20))  # 调整x坐标确保可见
            text_y = rect.bottom() - 2  # 将文本绘制在整个组件的底部边缘
            painter.drawText(text_x, text_y, start_text)
    
    def frame_to_x(self, frame, groove_rect):
        """将帧号转换为进度条上的x坐标"""
        total_frames = self.maximum() - self.minimum() + 1
        if total_frames == 0:
            return groove_rect.x()
        
        ratio = (frame - self.minimum()) / total_frames
        x = groove_rect.x() + ratio * groove_rect.width()
        return int(x)
    
    def x_to_frame(self, x, groove_rect):
        """将进度条上的x坐标转换为帧号"""
        total_frames = self.maximum() - self.minimum() + 1
        if total_frames == 0:
            return self.minimum()
        
        ratio = (x - groove_rect.x()) / groove_rect.width()
        frame = self.minimum() + ratio * total_frames
        return int(round(frame))
    
    def add_captured_segment(self, start_frame, end_frame):
        """添加一个捕获的片段"""
        self.captured_segments.append((start_frame, end_frame))
        self.update()
    
    def remove_captured_segment(self, index):
        """删除指定索引的捕获片段"""
        if 0 <= index < len(self.captured_segments):
            del self.captured_segments[index]
            self.update()
    
    def set_current_segment_start(self, frame):
        """设置当前正在捕获的片段的开始帧"""
        self.current_segment_start = frame
        self.update()
    
    def clear_current_segment_start(self):
        """清除当前正在捕获的片段的开始帧"""
        self.current_segment_start = None
        self.update()
    
    def clear_all_segments(self):
        """清除所有捕获的片段"""
        self.captured_segments.clear()
        self.current_segment_start = None
        self.update()
    
    def get_captured_segments(self):
        """获取所有捕获的片段"""
        return self.captured_segments.copy()
    
    def styleOption(self):
        """创建并返回一个QStyleOptionSlider对象"""
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        return option


class ClickableVideoLabel(QLabel):
    """可点击的视频标签，用于检测点击位置"""
    clicked = pyqtSignal(QPoint)  # 发送点击位置信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CrossCursor)  # 设置十字光标
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.pos())
        super().mousePressEvent(event)

import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.net(x)

# 模型类别配置
class_names = {
    0: 'no_boiling_over',
    1: 'boiling over',
}

class_colors = {
    'no_boiling_over': (255, 255, 255),  # 白色
    'boiling over': (0, 255, 0),         # 绿色
}

class ModelLoader:
    def __init__(self, model_type, model_path, num_classes, name=None):
        self.model_type = model_type
        self.model_path = model_path
        self.num_classes = num_classes
        self.name = name or os.path.basename(model_path)  # 使用文件名作为默认名称
        self.model = self.load_model()

    def load_model(self):
        if self.model_type == 'YOLO':
            try:
                return YOLO(self.model_path)
            except FileNotFoundError:
                print(f"检测模型文件不存在，请检查路径：{self.model_path}")
                return None
        elif self.model_type == 'EfficientNet':
            try:

                model = EfficientNet.from_name('efficientnet-b0', num_classes=self.num_classes)
                
                # num_ftrs = model._fc.in_features
                # model._fc = MLPBlock(input_dim=num_ftrs, hidden_dim=256, output_dim=2)

                # checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
                # #.pth文件读取
                # model.load_state_dict(checkpoint)
                # #.tar文件读取
                # model.load_state_dict(checkpoint['state_dict'])
                model.eval()
                return model
            except FileNotFoundError:
                print(f"分类模型文件不存在，请检查路径：{self.model_path}")
                return None
        else:
            print(f"不支持的模型类型：{self.model_type}")
            return None

class ModelManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("分类模型管理")
        self.setGeometry(200, 200, 600, 400)
        
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.SingleSelection)
        
        layout = QVBoxLayout()
        layout.addWidget(QLabel("已加载的分类模型:"))
        layout.addWidget(self.model_list)
        
        # 按钮布局
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("添加模型")
        self.add_btn.clicked.connect(self.add_model)
        btn_layout.addWidget(self.add_btn)
        
        self.remove_btn = QPushButton("删除模型")
        self.remove_btn.clicked.connect(self.remove_model)
        btn_layout.addWidget(self.remove_btn)
        
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.close_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def add_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择分类模型文件", 
            "", 
            "模型文件 (*.pth *.tar);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                model = EfficientNet.from_name('efficientnet-b0', num_classes=2)

                # num_ftrs = model._fc.in_features
                # model._fc = MLPBlock(input_dim=num_ftrs, hidden_dim=256, output_dim=2)

                checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
                
                # 检查状态字典格式
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                    
                model.load_state_dict(state_dict)
                model.eval()
                
                model_name = os.path.basename(file_path)
                self.model_list.addItem(model_name)
                self.parent().add_classification_model(model, model_name)
                
                QMessageBox.information(self, "成功", f"分类模型已加载: {model_name}")
            except Exception as e:
                QMessageBox.warning(self, "模型加载错误", f"无法加载分类模型: {str(e)}")
    
    def remove_model(self):
        selected = self.model_list.currentRow()
        if selected >= 0:
            model_name = self.model_list.item(selected).text()
            self.model_list.takeItem(selected)
            self.parent().remove_classification_model(model_name)
    
    def set_models(self, models):
        self.model_list.clear()
        for model_info in models:
            self.model_list.addItem(model_info['name'])

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频检测工具")
        self.setGeometry(100, 100, config.DEFAULT_WINDOW_SIZE[0], config.DEFAULT_WINDOW_SIZE[1])

        # 视频处理相关变量
        self.video_folder = ""
        self.video_files = []
        self.current_video_index = -1
        self.cap = None
        self.playing = False
        # 倍速功能已移除，固定使用1倍速播放
        self.current_frame_pos = 0
        self.total_frames = 0
        self.fps = config.DEFAULT_FPS
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.update_frame)
        
        # 模型推理相关变量
        self.model_loaded = False
        self.detection_model = None
        self.classification_models = []
        self.inference_results = None
        self.inference_active = False  # 新增：推理状态标志
        self.original_frame = None  # 新增：存储原始帧用于保存
        
        # 统计相关变量
        self.frame_statistics = {}  # 存储每帧的统计结果 {frame_num: {0: count, 1: count}}
        self.video_statistics = {0: 0, 1: 0}  # 整个视频的统计结果
        self.chart_save_dir = config.DEFAULT_CHART_DIR  # 图表保存目录
        
        # 自动保存相关变量
        self.previous_frame_confidence = None  # 上一帧的置信度
        self.previous_frame_class = None  # 上一帧的分类结果
        
        # 手动截取相关变量
        self.manual_capture_active = False  # 手动截取状态
        self.cut_segments = []  # 存储截取的时段 [(start_frame, end_frame), ...]
        self.current_segment_start = None  # 当前截取时段的起始帧
        self.saved_frames = {}  # 已保存的单帧字典，key为视频索引，value为保存的帧集合
        
        # 输出路径相关变量
        self.output_path = config.DEFAULT_OUTPUT_DIR  # 默认输出路径
        
        # 显示缩放相关变量（用于点击坐标转换）
        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0
        self.displayed_frame_size = (0, 0)  # 显示的帧尺寸
        
        # 锅具检测相关变量
        self.pot_detector = None  # 锅具检测器实例
        self.pot_model_path = None  # 锅具检测模型路径
        self.pot_detection_enabled = False  # 锅具检测状态
        self.pot_detection_save_dir = config.DEFAULT_POT_DETECTION_DIR  # 锅具检测结果保存目录
        
        # UI相关变量
        self.frame_info_label = QLabel()
        self.model_status_label = QLabel("模型未加载")
        self.model_status_label.setAlignment(Qt.AlignCenter)
        self.model_status_label.setStyleSheet("color: #FF5252; font-size: 12px;")
        self.is_seeking = False
        self.save_dir = config.DEFAULT_OUTPUT_DIR  # 添加保存目录

        # 按钮高亮状态
        self.active_button = None
        self.button_highlight_timer = QTimer(self)
        self.button_highlight_timer.timeout.connect(self.clear_button_highlight)
        self.button_highlight_duration = config.BUTTON_HIGHLIGHT_DURATION  # 高亮持续时间(毫秒)

        self.init_ui()
        
        # 设置窗口全屏显示
        self.showMaximized()
        
        # 尝试加载模型
        self.load_models()

    def init_ui(self):
        # 创建主控件和布局        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 顶部控制区域
        top_layout = QHBoxLayout()

        # 加载文件夹按钮
        self.load_folder_btn = QPushButton("加载视频文件夹")
        self.load_folder_btn.clicked.connect(self.load_video_folder)
        self.load_folder_btn.setFixedHeight(40)
        top_layout.addWidget(self.load_folder_btn)

        # 视频文件下拉框
        self.video_combo = QComboBox()
        self.video_combo.setFixedHeight(40)
        self.video_combo.currentIndexChanged.connect(self.select_video)
        top_layout.addWidget(self.video_combo, 2)

        # 添加模型加载按钮
        self.load_model_btn = QPushButton("加载检测模型")
        self.load_model_btn.setFixedHeight(40)
        self.load_model_btn.clicked.connect(self.load_model_dialog)
        top_layout.addWidget(self.load_model_btn)
        
        # 添加模型管理按钮
        self.manage_models_btn = QPushButton("管理分类模型")
        self.manage_models_btn.setFixedHeight(40)
        self.manage_models_btn.clicked.connect(self.manage_classification_models)
        top_layout.addWidget(self.manage_models_btn)

        # 添加设置输出路径按钮
        self.set_output_path_btn = QPushButton("设置输出路径")
        self.set_output_path_btn.setFixedHeight(40)
        self.set_output_path_btn.clicked.connect(self.set_output_path)
        # 设置按钮底色为橙色
        self.set_output_path_btn.setStyleSheet("background-color: brown; color: white;")
        top_layout.addWidget(self.set_output_path_btn)

        main_layout.addLayout(top_layout)

        # 视频显示区域
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)

        # 使用自定义的可点击标签
        self.video_label = ClickableVideoLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(config.VIDEO_LABEL_MIN_SIZE[0], config.VIDEO_LABEL_MIN_SIZE[1])
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.clicked.connect(self.on_video_clicked)  # 连接点击信号
        video_layout.addWidget(self.video_label)

        main_layout.addWidget(video_container, 5)

        # 控制按钮区域
        control_layout = QHBoxLayout()

        # 上一帧按钮
        self.prev_frame_btn = QPushButton("◀ 上一帧")
        self.prev_frame_btn.setFixedSize(100, 40)
        self.prev_frame_btn.clicked.connect(lambda: self.navigate_frames(-1))
        control_layout.addWidget(self.prev_frame_btn)

        # 快退5帧按钮
        self.prev_5_btn = QPushButton("◀◀ 快退5帧")
        self.prev_5_btn.setFixedSize(120, 40)
        self.prev_5_btn.clicked.connect(lambda: self.navigate_frames(-5))
        control_layout.addWidget(self.prev_5_btn)

        # 播放/暂停按钮
        self.play_btn = QPushButton("播放")
        self.play_btn.setFixedSize(80, 40)
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        control_layout.addWidget(self.play_btn)

        # 生成统计图表按钮
        self.generate_chart_btn = QPushButton("生成统计图表")
        self.generate_chart_btn.setFixedSize(150, 45)  # 增大按钮尺寸
        self.generate_chart_btn.clicked.connect(self.generate_statistics_chart)
        self.generate_chart_btn.setEnabled(False)
        control_layout.addWidget(self.generate_chart_btn)
        
        # 模型推理按钮 - 改为状态切换按钮
        self.inference_btn = QPushButton("开始推理")
        self.inference_btn.setFixedSize(100, 40)
        self.inference_btn.setCheckable(True)  # 设置为可切换状态
        self.inference_btn.clicked.connect(self.toggle_inference)
        self.inference_btn.setEnabled(False)
        control_layout.addWidget(self.inference_btn)
        
        # 手动捕获相关按钮
        control_layout.addStretch()  # 添加分隔
        
        # 帧跳转文本框
        self.frame_jump_edit = QLineEdit()
        self.frame_jump_edit.setFixedSize(120, 40)  # 与按钮大小匹配
        self.frame_jump_edit.setPlaceholderText("输入帧号")
        self.frame_jump_edit.returnPressed.connect(self.jump_to_frame)
        self.frame_jump_edit.setEnabled(False)
        control_layout.addWidget(self.frame_jump_edit)
        
        # 启用/禁用手动截取按钮
        self.manual_capture_btn = QPushButton("启用手动截取")
        self.manual_capture_btn.setFixedSize(150, 40)
        self.manual_capture_btn.setCheckable(True)  # 设置为可切换状态
        self.manual_capture_btn.clicked.connect(self.toggle_manual_capture)
        self.manual_capture_btn.setEnabled(False)
        control_layout.addWidget(self.manual_capture_btn)
        
        # 开始/结束截取时段按钮
        self.cut_segment_btn = QPushButton("开始截取")
        self.cut_segment_btn.setFixedSize(120, 40)
        self.cut_segment_btn.clicked.connect(self.toggle_cut_segment)
        self.cut_segment_btn.setEnabled(False)
        control_layout.addWidget(self.cut_segment_btn)
        
        # 导出捕获片段按钮

        
        # 删除捕获片段按钮
        self.delete_segment_btn = QPushButton("删除当前片段")
        self.delete_segment_btn.setFixedSize(150, 40)
        self.delete_segment_btn.clicked.connect(self.delete_current_segment)
        self.delete_segment_btn.setEnabled(False)
        control_layout.addWidget(self.delete_segment_btn)
        
        control_layout.addStretch()  # 添加分隔
        
        # 启用/禁用检测模型按钮
        self.toggle_detection_btn = QPushButton("启用检测模型")
        self.toggle_detection_btn.setFixedSize(150, 40)
        self.toggle_detection_btn.setCheckable(True)
        self.toggle_detection_btn.clicked.connect(self.toggle_detection_model)
        self.toggle_detection_btn.setEnabled(False)  # 只有加载了模型后才可用
        control_layout.addWidget(self.toggle_detection_btn)
        
        # 保存单帧按钮
        self.save_single_frame_btn = QPushButton("保存单帧")
        self.save_single_frame_btn.setFixedSize(150, 40)
        self.save_single_frame_btn.clicked.connect(self.save_single_frame)
        self.save_single_frame_btn.setEnabled(False)  # 只有视频加载后才可用
        control_layout.addWidget(self.save_single_frame_btn)
        
        # 分解截取片段按钮
        self.process_segments_btn = QPushButton("分解截取片段")
        self.process_segments_btn.setFixedSize(150, 40)
        self.process_segments_btn.clicked.connect(self.detect_and_crop_pots_in_segments)
        self.process_segments_btn.setEnabled(False)  # 只有有截取片段时才可用
        control_layout.addWidget(self.process_segments_btn)

        # 快进5帧按钮
        self.next_5_btn = QPushButton("快进5帧 ▶▶")
        self.next_5_btn.setFixedSize(120, 40)
        self.next_5_btn.clicked.connect(lambda: self.navigate_frames(5))
        control_layout.addWidget(self.next_5_btn)

        # 下一帧按钮
        self.next_frame_btn = QPushButton("下一帧 ▶")
        self.next_frame_btn.setFixedSize(100, 40)
        self.next_frame_btn.clicked.connect(lambda: self.navigate_frames(1))
        control_layout.addWidget(self.next_frame_btn)

        main_layout.addLayout(control_layout)

        # 倍速功能已移除

        # 进度条 - 使用自定义的CaptureSlider
        self.progress_bar = CaptureSlider(Qt.Horizontal)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.sliderMoved.connect(self.set_position)
        self.progress_bar.sliderPressed.connect(self.on_seek_start)
        self.progress_bar.sliderReleased.connect(self.on_seek_end)
        self.progress_bar.setEnabled(False)
        main_layout.addWidget(self.progress_bar)

        # 帧信息和模型状态
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.frame_info_label, alignment=Qt.AlignLeft)
        info_layout.addWidget(self.model_status_label, alignment=Qt.AlignRight)
        # 增加进度条和帧信息之间的间距
        main_layout.addSpacing(15)
        main_layout.addLayout(info_layout)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # 快捷键提示
        self.shortcut_label = QLabel(
            "快捷键: ←/→=快退/快进 | 空格=播放/暂停 | ↑/↓=切换视频 | A/D=单帧进退 | +=开始/结束截取 | *=保存单帧 | 推理后可生成统计图表"
        )
        self.shortcut_label.setAlignment(Qt.AlignCenter)
        self.shortcut_label.setStyleSheet("font-weight: bold; color: #555; padding: 2px;")  # 减少上下间距
        main_layout.addWidget(self.shortcut_label)

        # 初始化帧图像
        self.current_frame = None
        self.original_frame = None  # 确保初始化
        self.display_placeholder()

        # 禁用所有按钮的焦点获取
        self.disable_button_focus()

        # 初始按钮样式
        self.update_button_styles()

    def disable_button_focus(self):
        buttons = [
            self.play_btn, self.inference_btn, self.prev_frame_btn, self.next_frame_btn, 
            self.prev_5_btn, self.next_5_btn,
            self.load_model_btn, self.manage_models_btn, self.generate_chart_btn
        ]
        for btn in buttons:
            btn.setFocusPolicy(Qt.NoFocus)

    def update_button_styles(self):
        # 播放按钮样式
        if self.playing:
            self.play_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50; 
                    color: white;
                    border: 1px solid #388E3C;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #388E3C;
                }
            """)
        else:
            self.play_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f0f0f0; 
                    color: #333;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
            """)

        # 推理按钮样式
        if self.inference_active:
            self.inference_btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF5722; 
                    color: white;
                    border: 1px solid #E64A19;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #E64A19;
                }
            """)
        elif self.inference_btn.isEnabled():
            self.inference_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3; 
                    color: white;
                    border: 1px solid #0b7dda;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #0b7dda;
                }
            """)
        else:
            self.inference_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f0f0f0; 
                    color: #333;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
            """)

        # 模型加载按钮样式
        self.load_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0; 
                color: white;
                border: 1px solid #7B1FA2;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        
        # 模型管理按钮样式
        self.manage_models_btn.setStyleSheet("""
            QPushButton {
                background-color: #009688; 
                color: white;
                border: 1px solid #00796B;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #00796B;
            }
        """)
        
        # 生成统计图表按钮样式
        self.generate_chart_btn.setStyleSheet("""
            QPushButton {
                background-color: #795548; 
                color: white;
                border: 1px solid #5D4037;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5D4037;
            }
        """)

        # 导航按钮样式
        nav_buttons = [self.prev_frame_btn, self.next_frame_btn, self.prev_5_btn, self.next_5_btn]
        for btn in nav_buttons:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #607D8B; 
                    color: white;
                    border: 1px solid #546E7A;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #546E7A;
                }
            """)




        # 倍速按钮样式已移除

        # 高亮当前活动按钮
        if self.active_button:
            self.active_button.setStyleSheet("""
                QPushButton {
                    background-color: #FF5722; 
                    color: white;
                    border: 2px solid #E64A19;
                    border-radius: 4px;
                    font-weight: bold;
                }
            """)

    def display_placeholder(self):
        pixmap = QPixmap(800, 400)
        pixmap.fill(QColor(30, 30, 30))
        painter = QPainter(pixmap)
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont("Arial", 24))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "加载视频文件夹开始使用")
        painter.end()
        self.video_label.setPixmap(pixmap)

    def load_video_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择视频文件夹")

        if folder_path:
            self.video_folder = folder_path
            self.video_files = []
            self.video_combo.clear()
            
            # 禁用控件
            self.frame_jump_edit.setEnabled(False)

            # 扫描支持的视频文件
            extensions = config.SUPPORTED_VIDEO_EXTENSIONS
            for file in os.listdir(folder_path):
                if os.path.splitext(file)[1].lower() in extensions:
                    self.video_files.append(file)

            if not self.video_files:
                QMessageBox.warning(self, "无视频文件", "该文件夹中没有找到支持的视频文件")
                return

            self.video_combo.addItems(self.video_files)
            self.status_bar.showMessage(f"已加载文件夹: {folder_path} | 共 {len(self.video_files)} 个视频")
            self.highlight_button(self.load_folder_btn)

    def select_video(self, index):
        if index < 0 or index >= len(self.video_files):
            return

        self.current_video_index = index
        video_file = os.path.join(self.video_folder, self.video_files[index])
        
        # 重置统计数据
        self.frame_statistics = {}
        self.video_statistics = {0: 0, 1: 0}
        self.generate_chart_btn.setEnabled(False)

        # 释放之前的视频资源
        if self.cap:
            self.cap.release()
            self.frame_timer.stop()
            # 立即重置所有帧数相关变量和显示
            self.current_frame_pos = 0
            self.total_frames = 0
            self.frame_info_label.setText("当前帧：0 / 总帧数：0")
            # 清除进度条上的所有捕获片段和标记
            self.progress_bar.clear_all_segments()
            # 清除VideoPlayer类中的截取片段和状态
            self.cut_segments.clear()
            self.current_segment_start = None
            # 重置截取按钮状态
            self.cut_segment_btn.setText("开始截取")
            self.delete_segment_btn.setEnabled(False)
            self.process_segments_btn.setEnabled(False)

        # 初始化视频捕获
        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", f"无法打开视频文件: {video_file}")
            return

        # 获取总帧数 - 改进的方法：先尝试使用CAP_PROP_FRAME_COUNT，如果返回0则逐帧计数
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_pos = 0
        self.playing = False
        self.play_btn.setText("播放")
        
        # 如果CAP_PROP_FRAME_COUNT返回0，尝试逐帧计数
        if self.total_frames == 0:
            # 使用更可靠的方法获取总帧数
            # 创建一个新的视频捕获对象来计数，避免影响原始流
            temp_cap = cv2.VideoCapture(video_file)
            self.total_frames = 0
            # 逐帧读取直到结束
            while True:
                ret, _ = temp_cap.read()
                if not ret:
                    break
                self.total_frames += 1
            # 释放临时视频捕获对象
            temp_cap.release()
            # 如果仍然为0，设置为1作为默认值
            if self.total_frames == 0:
                self.total_frames = 1

        # 设置进度条范围
        self.progress_bar.setRange(0, self.total_frames - 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setEnabled(True)
        
        # 启用控件
        self.play_btn.setEnabled(True)
        self.inference_btn.setEnabled(self.model_loaded)
        self.prev_frame_btn.setEnabled(True)
        self.next_frame_btn.setEnabled(True)
        self.prev_5_btn.setEnabled(True)
        self.next_5_btn.setEnabled(True)
        self.manual_capture_btn.setEnabled(True)
        self.frame_jump_edit.setEnabled(True)
        self.save_single_frame_btn.setEnabled(True)

        # 显示第一帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.original_frame = frame.copy()  # 保存原始帧
            self.display_frame(frame)
            # 确保显示正确的帧数
            self.current_frame_pos = 0
            self.frame_info_label.setText(f"当前帧：{self.current_frame_pos + 1} / 总帧数：{self.total_frames}")
        self.status_bar.showMessage(
            f"已加载: {self.video_files[index]} | 总帧数: {self.total_frames} | FPS: {self.fps:.2f}")

    def toggle_play(self):
        if not self.cap or not self.cap.isOpened():
            return

        self.playing = not self.playing

        if self.playing:
            self.play_btn.setText("暂停")
            self.frame_timer.start(int(1000 / self.fps))  # 固定1倍速播放
        else:
            self.play_btn.setText("播放")
            self.frame_timer.stop()

        self.update_button_styles()
        self.setFocus()
        self.highlight_button(self.play_btn)

    # set_speed函数已移除，倍速功能不再需要

    def set_position(self, position):
        if not self.cap:
            return
        position = max(0, min(position, self.total_frames - 1))
        if self.current_frame_pos == position:
            return
            
        # 关键修复：确保current_frame_pos和实际视频位置一致
        self.current_frame_pos = position
        
        try:
            # 无论是否正在播放，都设置视频位置
            # 这样可以确保拖动进度条后视频从正确位置开始播放
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_pos)
            
            if not self.playing:
                # 读取当前位置的帧
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    self.current_frame = frame
                    self.original_frame = frame.copy()  # 更新原始帧
                    # 异步处理推理，避免阻塞UI
                    if self.inference_active:
                        QTimer.singleShot(0, lambda: self.run_inference(frame))
                    self.display_frame(frame)
                    self.frame_info_label.setText(f"当前帧：{self.current_frame_pos + 1} / 总帧数：{self.total_frames}")
                else:
                    # 如果读取失败，尝试重新设置位置
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_pos)
        except Exception as e:
            # 捕获所有异常，避免应用卡死
            print(f"Error in set_position: {e}")
        
        self.progress_bar.setValue(self.current_frame_pos)
        self.setFocus()

    def jump_to_frame(self):
        """根据文本框输入的帧号跳转到对应帧"""
        if not self.cap or not self.cap.isOpened():
            return
        
        try:
            frame_num = int(self.frame_jump_edit.text()) - 1  # 转换为0-based索引
            frame_num = max(0, min(frame_num, self.total_frames - 1))  # 确保在有效范围内
            
            if frame_num != self.current_frame_pos:
                # 先暂停播放
                if self.playing:
                    self.toggle_play()
                
                # 设置位置
                self.set_position(frame_num)
                
                # 清空文本框
                self.frame_jump_edit.clear()
                
                self.highlight_button(self.frame_jump_edit)  # 高亮显示操作
        except ValueError:
            # 如果输入不是数字，显示错误提示
            QMessageBox.warning(self, "输入错误", "请输入有效的帧号")
            self.frame_jump_edit.clear()
            self.frame_jump_edit.setFocus()

    def on_seek_start(self):
        self.is_seeking = True
        if self.playing:
            self.was_playing = True
            self.toggle_play()
        else:
            self.was_playing = False

    def on_seek_end(self):
        self.is_seeking = False
        if self.was_playing:
            self.toggle_play()

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            if self.playing and not self.is_seeking:
                try:
                    # 关键修复：先检查当前位置是否超出总帧数
                    if self.current_frame_pos >= self.total_frames:
                        # 视频结束，重置到第一帧
                        self.current_frame_pos = 0
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.toggle_play()
                        if self.inference_active and self.frame_statistics:
                            QTimer.singleShot(0, self.generate_statistics_chart)
                            self.status_bar.showMessage("视频播放完成，统计图表已自动生成")
                        return
                    
                    # 确保视频位置正确
                    current_cap_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if abs(current_cap_pos - self.current_frame_pos) > 1:
                        # 如果位置偏差太大，重新设置位置
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_pos)
                    
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        # 手动递增current_frame_pos，而不是从cap.get获取
                        # 这样可以确保位置的准确性，避免视频越来越短
                        self.current_frame_pos += 1
                        self.current_frame_pos = min(self.current_frame_pos, self.total_frames - 1)
                        self.current_frame = frame
                        self.original_frame = frame.copy()  # 更新原始帧
                        self.progress_bar.setValue(self.current_frame_pos)
                        
                        # 异步处理推理，避免阻塞UI
                        if self.inference_active:
                            # 使用QTimer.singleShot延迟执行推理，避免UI卡死
                            QTimer.singleShot(0, lambda: self.run_inference(frame))
                            
                        self.display_frame(frame)
                        self.frame_info_label.setText(f"当前帧：{self.current_frame_pos + 1} / 总帧数：{self.total_frames}")
                    else:
                        # 视频结束，重置到第一帧
                        self.current_frame_pos = 0
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.toggle_play()
                        
                        # 如果处于推理状态，自动生成统计图表
                        if self.inference_active and self.frame_statistics:
                            # 异步生成图表，避免阻塞UI
                            QTimer.singleShot(0, self.generate_statistics_chart)
                            self.status_bar.showMessage("视频播放完成，统计图表已自动生成")
                except Exception as e:
                    # 捕获所有异常，避免应用卡死
                    print(f"Error in update_frame: {e}")
                    # 停止播放，避免无限循环错误
                    self.toggle_play()

    def display_frame(self, frame):
        if frame is None:
            return

        # 保存原始帧尺寸用于坐标转换
        original_h, original_w = frame.shape[:2]

        # 如果有推理结果且处于推理状态，将其绘制到帧上
        if self.inference_active and self.inference_results is not None:
            frame = self.draw_inference_results(frame)

        # 将OpenCV图像转换为Qt图像
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # 缩放以适应标签大小
        label_size = self.video_label.size()
        scaled_pixmap = pixmap.scaled(
            label_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # 计算缩放比例和偏移量（用于点击坐标转换）
        scaled_w = scaled_pixmap.width()
        scaled_h = scaled_pixmap.height()
        
        self.display_scale = min(scaled_w / original_w, scaled_h / original_h)
        self.displayed_frame_size = (original_w, original_h)
        
        # 计算偏移量（图像居中显示时的边距）
        self.display_offset_x = (label_size.width() - scaled_w) // 2
        self.display_offset_y = (label_size.height() - scaled_h) // 2
        
        self.video_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_frame is not None:
            self.display_frame(self.current_frame)

    def keyPressEvent(self, event):
        if not self.cap:
            return super().keyPressEvent(event)

        key = event.key()
        step = 0

        # 按键映射，只保留导航功能
        key_map = {
            Qt.Key_Left: -5,
            Qt.Key_Right: 5,
            Qt.Key_A: -1,
            Qt.Key_D: 1,
            Qt.Key_Space: None,
            Qt.Key_Up: None,
            Qt.Key_Down: None,
            Qt.Key_Plus: None,  # 小键盘加号键 - 开始/结束捕获快捷键
            Qt.Key_Equal: None,  # 主键盘等号/加号键 - 开始/结束捕获快捷键
            Qt.Key_Asterisk: None  # 小键盘星号键 - 保存单帧快捷键
        }
        
        action = key_map.get(key, None)

        if action is not None:
            # 处理导航操作
            self.navigate_frames(action)
            # 高亮对应的导航按钮
            if key == Qt.Key_Left:
                self.highlight_button(self.prev_5_btn)
            elif key == Qt.Key_Right:
                self.highlight_button(self.next_5_btn)
            elif key == Qt.Key_A:
                self.highlight_button(self.prev_frame_btn)
            elif key == Qt.Key_D:
                self.highlight_button(self.next_frame_btn)
        elif key == Qt.Key_Space:
            self.toggle_play()
        elif key == Qt.Key_Up and self.current_video_index > 0:
            self.video_combo.setCurrentIndex(self.current_video_index - 1)
        elif key == Qt.Key_Down and self.current_video_index < len(self.video_files) - 1:
            self.video_combo.setCurrentIndex(self.current_video_index + 1)
        elif key == Qt.Key_Plus or (key == Qt.Key_Equal and event.modifiers() == Qt.ShiftModifier):
            # 触发开始/结束捕获（同时支持小键盘加号和主键盘Shift++）
            self.toggle_cut_segment()
            self.highlight_button(self.cut_segment_btn)
        elif key == Qt.Key_Asterisk:
            # 触发保存单帧功能（小键盘星号键）
            self.save_single_frame()
            self.highlight_button(self.save_single_frame_btn)
        else:
            super().keyPressEvent(event)

        self.setFocus()

    # 处理鼠标滚轮事件
    def wheelEvent(self, event):
        if not self.cap or not self.cap.isOpened():
            return super().wheelEvent(event)
            
        # 获取滚轮滚动方向和步数
        delta = event.angleDelta().y()
        step = 5 if delta > 0 else -5  # 向上滚动前进5帧，向下滚动后退5帧
        
        # 暂停播放并导航帧
        if self.playing:
            self.toggle_play()
            
        self.navigate_frames(step)
        
        # 高亮对应的导航按钮
        self.highlight_button(self.prev_5_btn if step < 0 else self.next_5_btn)

    def navigate_frames(self, step):
        if not self.cap:
            return
            
        # 导航时暂停播放
        if self.playing:
            self.toggle_play()
            
        new_pos = self.current_frame_pos + step
        new_pos = max(0, min(new_pos, self.total_frames - 1))
        
        if new_pos != self.current_frame_pos:
            self.set_position(new_pos)

    def toggle_inference(self):
        if not self.model_loaded:
            QMessageBox.warning(self, "模型未加载", "请确保检测模型和分类模型文件存在于正确路径")
            self.inference_btn.setChecked(False)  # 确保按钮状态正确
            return

        # 切换推理状态
        self.inference_active = self.inference_btn.isChecked()
        
        if self.inference_active:
            self.inference_btn.setText("停止推理")
            self.status_bar.showMessage("模型推理已启用，正在处理当前帧...")
            if self.current_frame is not None:
                self.run_inference(self.current_frame)
                self.display_frame(self.current_frame)
        else:
            self.inference_btn.setText("开始推理")
            self.status_bar.showMessage("模型推理已禁用")
            self.inference_results = None
            # 刷新显示原始帧
            if self.current_frame is not None:
                self.display_frame(self.current_frame)
                
        self.update_button_styles()
        self.highlight_button(self.inference_btn)
    
    def toggle_manual_capture(self):
        """启用/禁用手动截取功能"""
        # 切换手动捕获状态
        self.manual_capture_active = self.manual_capture_btn.isChecked()
        
        if self.manual_capture_active:
            self.manual_capture_btn.setText("禁用手动截取")
            self.cut_segment_btn.setEnabled(True)
            self.status_bar.showMessage("手动截取功能已启用，点击'开始截取'按钮开始截取时段")
        else:
            self.manual_capture_btn.setText("启用手动截取")
            self.cut_segment_btn.setEnabled(False)
            self.cut_segment_btn.setText("开始截取")
            self.current_segment_start = None
            self.status_bar.showMessage("手动截取功能已禁用")
        
        # 如果有捕获的时段，启用处理按钮（无需模型加载）
        if self.cut_segments:
            self.process_segments_btn.setEnabled(True)
        else:
            self.process_segments_btn.setEnabled(False)
        
        self.update_button_styles()
        self.highlight_button(self.manual_capture_btn)
    
    def toggle_cut_segment(self):
        """开始/结束截取一个时段"""
        if not self.manual_capture_active:
            QMessageBox.warning(self, "手动截取未启用", "请先启用手动截取功能")
            return
        
        if self.current_segment_start is None:
            # 开始截取一个新的时段
            self.current_segment_start = self.current_frame_pos
            self.progress_bar.set_current_segment_start(self.current_segment_start)
            self.cut_segment_btn.setText("结束截取")
            self.status_bar.showMessage(f"开始截取时段，起始帧: {self.current_segment_start + 1}")
        else:
            # 结束当前时段的截取
            end_frame = self.current_frame_pos
            
            # 确保起始帧小于结束帧
            if self.current_segment_start < end_frame:
                self.cut_segments.append((self.current_segment_start, end_frame))
                self.progress_bar.add_captured_segment(self.current_segment_start, end_frame)
                self.progress_bar.clear_current_segment_start()
                self.delete_segment_btn.setEnabled(True)  # 启用删除按钮
                
                # 有截取片段时启用处理按钮（无需模型加载）
                self.process_segments_btn.setEnabled(True)
                    
                self.status_bar.showMessage(f"已截取时段: 帧 {self.current_segment_start + 1} 到 {end_frame + 1}")
            else:
                QMessageBox.warning(self, "截取错误", "结束帧必须大于起始帧")
            
            # 重置当前时段起始帧
            self.current_segment_start = None
            self.progress_bar.clear_current_segment_start()
            self.cut_segment_btn.setText("开始截取")
        
        self.highlight_button(self.cut_segment_btn)
    
    def delete_current_segment(self):
        """删除包含当前帧位置的截取片段"""
        if not self.cut_segments:
            QMessageBox.warning(self, "无截取片段", "没有截取任何片段")
            return
        
        # 查找包含当前帧的片段
        segment_index = -1
        target_segment = None
        for i, (start_frame, end_frame) in enumerate(self.cut_segments):
            if start_frame <= self.current_frame_pos <= end_frame:
                segment_index = i
                target_segment = (start_frame, end_frame)
                break
        
        if segment_index == -1:
            QMessageBox.warning(self, "未找到片段", "当前位置不在任何截取的片段内")
            return
        
        # 删除找到的片段
        del self.cut_segments[segment_index]
        
        # 在progress_bar的captured_segments中找到对应的片段并删除
        # 遍历captured_segments列表，找到匹配的片段
        for i, (start, end) in enumerate(self.progress_bar.captured_segments):
            if start == target_segment[0] and end == target_segment[1]:
                self.progress_bar.remove_captured_segment(i)
                break
        
        # 更新UI状态
        if not self.cut_segments:
            self.process_segments_btn.setEnabled(False)
            self.delete_segment_btn.setEnabled(False)
        
        self.status_bar.showMessage(f"已删除片段 {segment_index + 1}")
        self.highlight_button(self.delete_segment_btn)

    def load_model_dialog(self):
        """打开文件对话框加载检测模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择检测模型文件", 
            "", 
            "YOLO模型文件 (*.pt);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
            
        # 加载检测模型
        try:
            # 加载主检测模型
            self.detection_model = YOLO(file_path)
            
            # 同时加载锅具检测模型
            self.pot_model_path = file_path
            self.pot_detector = PotDetector(model_path=file_path)
            
            if self.pot_detector.model_loaded:
                self.status_bar.showMessage(f"检测模型已加载: {os.path.basename(file_path)}")
                self.highlight_button(self.load_model_btn)
                
                # 检查模型状态
                self.check_model_status()
                
                # 启用检测模型按钮
                self.toggle_detection_btn.setEnabled(True)
                
                # 自动启用检测模型功能
                self.toggle_detection_btn.setChecked(True)
                self.toggle_detection_model()
            else:
                QMessageBox.warning(self, "模型加载失败", f"无法加载检测模型: {file_path}")
                self.pot_detector = None
                self.pot_model_path = None
                self.model_loaded = False
        except Exception as e:
            QMessageBox.warning(self, "模型加载错误", f"无法加载检测模型: {str(e)}")
            self.pot_detector = None
            self.pot_model_path = None
            self.model_loaded = False

    def manage_classification_models(self):
        """打开分类模型管理对话框"""
        dialog = ModelManagerDialog(self)
        dialog.set_models(self.classification_models)
        dialog.exec_()
    
    def add_classification_model(self, model, model_name):
        """添加一个新的分类模型"""
        self.classification_models.append({
            'model': model,
            'name': model_name
        })
        
        # 检查模型状态
        self.check_model_status()
        
        # 如果当前正在推理，重新运行推理
        if self.inference_active and self.current_frame is not None:
            self.run_inference(self.current_frame)
            self.display_frame(self.current_frame)
    
    def remove_classification_model(self, model_name):
        """删除指定的分类模型"""
        # 找到并删除模型
        for i, model_info in enumerate(self.classification_models):
            if model_info['name'] == model_name:
                del self.classification_models[i]
                break
        
        # 检查模型状态
        self.check_model_status()
        
        # 如果当前正在推理，重新运行推理
        if self.inference_active and self.current_frame is not None:
            self.run_inference(self.current_frame)
            self.display_frame(self.current_frame)
    
    def check_model_status(self):
        """检查模型状态并更新UI"""
        if self.detection_model is not None and self.classification_models:
            model_names = ", ".join([model['name'] for model in self.classification_models])
            self.model_loaded = True
            self.model_status_label.setText(f"模型已加载 ({model_names})")
            self.model_status_label.setStyleSheet("color: #4CAF50; font-size: 12px;")
            self.inference_btn.setEnabled(True)
        else:
            self.model_loaded = False
            if self.detection_model is None:
                self.model_status_label.setText("检测模型未加载")
            elif not self.classification_models:
                self.model_status_label.setText("分类模型未加载")
            else:
                self.model_status_label.setText("模型未加载")
                
            self.model_status_label.setStyleSheet("color: #FF5252; font-size: 12px;")
            self.inference_btn.setEnabled(False)
            
            # 如果推理按钮被按下，重置状态
            if self.inference_active:
                self.inference_active = False
                self.inference_btn.setChecked(False)
                self.inference_btn.setText("开始推理")

    def load_models(self):
        """加载默认检测模型（如果存在）"""
        # 检查默认模型文件是否存在
        detection_path = config.DEFAULT_DETECTION_MODEL
        
        # 加载检测模型
        if os.path.exists(detection_path):
            try:
                self.detection_model = YOLO(detection_path)
                self.model_loaded = True
                self.model_status_label.setText("默认检测模型已加载")
                self.model_status_label.setStyleSheet("color: #4CAF50; font-size: 12px;")
            except Exception as e:
                print(f"无法加载检测模型: {str(e)}")
                self.model_loaded = False

    def apply_ring_mask(self, image):
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        radius = max(height, width) // 2

        radius_inner = int(radius * config.RING_MASK_INNER_RADIUS_RATIO)
        radius_outer = int(radius * config.RING_MASK_OUTER_RADIUS_RATIO)

        outer_circle = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(outer_circle, center, radius + radius_outer, 255, -1)

        inner_circle = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(inner_circle, center, radius - radius_inner, 255, -1)

        ring_mask = cv2.subtract(outer_circle, inner_circle)
        ring_mask = np.uint8(ring_mask)  # 确保掩膜是uint8

        # 应用掩膜：只保留圆环区域
        result = cv2.bitwise_and(image, image, mask=ring_mask)
        return result
    def run_inference(self, frame):
        """对当前帧运行模型推理"""
        if not self.model_loaded or frame is None:
            return None

        try:
            # 保存原始帧（用于后续保存操作）
            self.original_frame = frame.copy()  # 关键修改：保存原始帧
            
            # 进行目标检测
            detection_results = self.detection_model(frame)
            
            results = []
            for result in detection_results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # 扩大检测框
                    scale_factor = 1.05
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
                    # x1_new = x1_new+(x2_new-x1_new) // 8
                    # 提取并预处理图像
                    cropped_image = frame[y1_new:y2_new, x1_new : x2_new]

                    # 将裁剪图直接拉伸为 224x224，不补黑边
                    image = cv2.resize(cropped_image, (224, 224), interpolation=cv2.INTER_LINEAR)

                    # [h, w, _] = cropped_image.shape
                    #
                    # if h == 0 or w == 0:
                    #     continue
                    #
                    # length = max((h, w))
                    # image = np.zeros((length, length, 3), np.uint8)
                    # image[0:h, 0:w] = cropped_image

                    # 添加：圆环预处理
                    image = self.apply_ring_mask(image)

                    # 准备模型输入
                    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(224, 224), swapRB=True)
                    efficientnet_input = torch.from_numpy(blob)

                    # 对每个分类模型进行推理
                    model_results = []
                    for model_info in self.classification_models:
                        model = model_info['model']
                        model_name = model_info['name']

                        with torch.no_grad():
                            classification_results = model(efficientnet_input)
                            probabilities = torch.softmax(classification_results, dim=1)
                            max_prob, predicted_indices = torch.max(probabilities, dim=1)

                            predicted_class_index = predicted_indices.item()
                            confidence = max_prob.item()
                            predicted_class = class_names.get(predicted_class_index, f"未知类别({predicted_class_index})")

                            model_results.append({
                                'model_name': model_name,
                                'class': predicted_class,
                                'class_index': predicted_class_index,
                                'confidence': confidence
                            })

                    # 合并所有模型的结果（使用投票或平均）
                    if model_results:
                        # 简单投票：选择出现次数最多的类别
                        class_counts = {}
                        for result in model_results:
                            cls = result['class_index']
                            class_counts[cls] = class_counts.get(cls, 0) + 1

                        # 找到票数最多的类别
                        voted_class = max(class_counts, key=class_counts.get)
                        voted_class_name = class_names.get(voted_class, f"未知类别({voted_class})")

                        # 计算平均置信度
                        avg_confidence = sum(r['confidence'] for r in model_results) / len(model_results)

                        # 记录统计信息
                        if self.current_frame_pos not in self.frame_statistics:
                            self.frame_statistics[self.current_frame_pos] = {0: 0, 1: 0}
                        self.frame_statistics[self.current_frame_pos][voted_class] += 1
                        self.video_statistics[voted_class] += 1

                        # 保存推理结果
                        results.append({
                            'box': (x1, y1, x2, y2),
                            'class': voted_class_name,
                            'class_index': voted_class,
                            'confidence': avg_confidence,
                            'model_results': model_results
                        })

            self.inference_results = results

            # 启用生成图表按钮
            if self.frame_statistics:
                self.generate_chart_btn.setEnabled(True)

            return results
        except Exception as e:
            print(f"推理错误: {str(e)}")
            return None

    def draw_inference_results(self, frame):
        """在帧上绘制推理结果"""
        if not self.inference_results:
            return frame

        for result in self.inference_results:
            x1, y1, x2, y2 = result['box']
            label = f"{result['class']}: {result['confidence']:.2f}"
            color = class_colors.get(result['class'], (0, 255, 0))

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # 绘制标签
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return frame

    def generate_statistics_chart(self):
        """生成统计图表"""
        if not self.frame_statistics:
            QMessageBox.warning(self, "无统计数据", "没有检测到任何数据，请先运行推理")
            return

        # 确保图表保存目录存在
        os.makedirs(self.chart_save_dir, exist_ok=True)

        # 准备数据
        frames = sorted(self.frame_statistics.keys())
        class_0_counts = []
        class_1_counts = []

        for frame in frames:
            stats = self.frame_statistics[frame]
            class_0_counts.append(stats.get(0, 0))
            class_1_counts.append(stats.get(1, 0))

        # 创建图表
        plt.figure(figsize=(12, 6))

        # 绘制折线图
        plt.plot(frames, class_0_counts, label='无沸腾', color='blue')
        plt.plot(frames, class_1_counts, label='有沸腾', color='green')

        # 添加标题和标签
        plt.title('沸腾检测统计')
        plt.xlabel('帧序号')
        plt.ylabel('检测次数')
        plt.legend()
        plt.grid(True)

        # 保存图表
        video_name = os.path.basename(self.video_files[self.current_video_index]) if self.video_files else 'unknown'
        chart_filename = f"{os.path.splitext(video_name)[0]}_stats.png"
        chart_path = os.path.join(self.chart_save_dir, chart_filename)

        try:
            plt.savefig(chart_path)
            plt.close()
            
            # 显示成功消息
            QMessageBox.information(
                self, 
                "图表生成成功", 
                f"统计图表已保存到:\n{chart_path}\n\n" 
                f"总帧数: {len(frames)}\n" 
                f"无沸腾: {self.video_statistics.get(0, 0)} 帧\n" 
                f"有沸腾: {self.video_statistics.get(1, 0)} 帧"
            )
            
            # 高亮显示操作
            self.highlight_button(self.generate_chart_btn)
        except Exception as e:
            QMessageBox.warning(self, "图表生成失败", f"无法生成统计图表: {str(e)}")

    def save_single_frame(self):
        """保存当前帧"""
        if not self.cap or not self.cap.isOpened():
            return

        # 确保输出目录存在
        os.makedirs(self.output_path, exist_ok=True)

        # 确保有原始帧可用
        if self.original_frame is None:
            self.original_frame = self.current_frame.copy()

        # 生成文件名
        video_name = os.path.basename(self.video_files[self.current_video_index]) if self.video_files else 'unknown'
        frame_filename = f"{os.path.splitext(video_name)[0]}_frame_{self.current_frame_pos + 1}.png"
        frame_path = os.path.join(self.output_path, frame_filename)

        try:
            # 保存原始帧（不包含任何绘制的检测结果）
            cv2.imwrite(frame_path, self.original_frame)
            
            # 更新已保存的帧集合
            if self.current_video_index not in self.saved_frames:
                self.saved_frames[self.current_video_index] = set()
            self.saved_frames[self.current_video_index].add(self.current_frame_pos)

            # 显示成功消息
            QMessageBox.information(self, "保存成功", f"当前帧已保存到:\n{frame_path}")
            
            # 高亮显示操作
            self.highlight_button(self.save_single_frame_btn)
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"无法保存当前帧: {str(e)}")

    def detect_and_crop_pots_in_segments(self):
        """对截取的时段进行锅具检测和裁剪"""
        if not self.cut_segments:
            QMessageBox.warning(self, "无截取片段", "没有截取任何片段")
            return

        if not self.pot_detector or not self.pot_detector.model_loaded:
            QMessageBox.warning(self, "锅具检测器未加载", "请先加载检测模型")
            return

        # 确保输出目录存在
        os.makedirs(self.pot_detection_save_dir, exist_ok=True)

        # 处理每个截取的时段
        for i, (start_frame, end_frame) in enumerate(self.cut_segments):
            # 创建时段专用的输出目录
            segment_dir = os.path.join(self.pot_detection_save_dir, f"segment_{i+1}_frames_{start_frame+1}_{end_frame+1}")
            os.makedirs(segment_dir, exist_ok=True)

            # 重新打开视频文件
            video_file = os.path.join(self.video_folder, self.video_files[self.current_video_index])
            temp_cap = cv2.VideoCapture(video_file)
            
            if not temp_cap.isOpened():
                QMessageBox.warning(self, "视频打开失败", f"无法打开视频文件: {video_file}")
                continue

            # 设置起始帧
            temp_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 处理时段内的每一帧
            current_frame = start_frame
            saved_count = 0
            
            while current_frame <= end_frame:
                ret, frame = temp_cap.read()
                if not ret:
                    break

                # 进行锅具检测和裁剪
                cropped_images, detections = self.pot_detector.detect_and_crop_pots(frame)

                # 保存裁剪出的锅具图像
                for j, cropped_img in enumerate(cropped_images):
                    # 生成文件名
                    frame_filename = f"frame_{current_frame+1}_pot_{j+1}.png"
                    frame_path = os.path.join(segment_dir, frame_filename)

                    # 保存图像
                    cv2.imwrite(frame_path, cropped_img)
                    saved_count += 1

                current_frame += 1

            # 释放资源
            temp_cap.release()

            # 显示处理结果
            self.status_bar.showMessage(f"已处理片段 {i+1}/{len(self.cut_segments)}，保存了 {saved_count} 个锅具图像")

        # 显示总处理结果
        QMessageBox.information(
            self, 
            "处理完成", 
            f"已处理所有截取片段，结果保存到:\n{self.pot_detection_save_dir}"
        )

    def toggle_detection_model(self):
        """启用/禁用检测模型"""
        self.pot_detection_enabled = self.toggle_detection_btn.isChecked()
        
        if self.pot_detection_enabled:
            self.toggle_detection_btn.setText("禁用检测模型")
            self.status_bar.showMessage("检测模型已启用")
        else:
            self.toggle_detection_btn.setText("启用检测模型")
            self.status_bar.showMessage("检测模型已禁用")

    def set_output_path(self):
        """设置输出路径"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择输出路径")
        if folder_path:
            self.output_path = folder_path
            self.pot_detection_save_dir = os.path.join(folder_path, "pot_detection")
            self.chart_save_dir = os.path.join(folder_path, "chart")
            self.status_bar.showMessage(f"输出路径已设置为: {folder_path}")
            self.highlight_button(self.set_output_path_btn)

    def highlight_button(self, button):
        """高亮显示按钮"""
        self.active_button = button
        self.update_button_styles()
        
        # 设置定时器，一段时间后清除高亮
        self.button_highlight_timer.stop()
        self.button_highlight_timer.start(self.button_highlight_duration)

    def clear_button_highlight(self):
        """清除按钮高亮"""
        self.active_button = None
        self.update_button_styles()

    def on_video_clicked(self, pos):
        """处理视频点击事件"""
        # 计算实际点击位置在原始帧上的坐标
        scaled_x = (pos.x() - self.display_offset_x) / self.display_scale
        scaled_y = (pos.y() - self.display_offset_y) / self.display_scale
        
        # 确保坐标在有效范围内
        scaled_x = max(0, min(scaled_x, self.displayed_frame_size[0] - 1))
        scaled_y = max(0, min(scaled_y, self.displayed_frame_size[1] - 1))
        
        # 显示点击位置信息
        self.status_bar.showMessage(f"点击位置: ({int(scaled_x)}, {int(scaled_y)}) (原始帧坐标)")

if __name__ == '__main__':
    # app = QApplication(sys.argv)
    # player = VideoPlayer()
    # sys.exit(app.exec_())

    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    # 设置全局字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)

    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())