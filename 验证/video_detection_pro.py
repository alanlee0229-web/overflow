import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ultralytics import YOLO
import torch
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
import matplotlib
# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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
        self.setGeometry(100, 100, 1000, 700)

        # 视频处理相关变量
        self.video_folder = ""
        self.video_files = []
        self.current_video_index = -1
        self.cap = None
        self.playing = False
        # 倍速功能已移除，固定使用1倍速播放
        self.current_frame_pos = 0
        self.total_frames = 0
        self.fps = 30
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
        self.chart_save_dir = os.path.join(os.getcwd(), "chart")  # 图表保存目录
        
        # 自动保存相关变量
        self.images_save_dir = os.path.join(os.getcwd(), "images")  # 已溢图片保存目录
        self.problem_save_dir = os.path.join(os.getcwd(), "problem")  # 问题图片保存目录
        self.previous_frame_confidence = None  # 上一帧的置信度
        self.previous_frame_class = None  # 上一帧的分类结果
        
        # UI相关变量
        self.frame_info_label = QLabel()
        self.model_status_label = QLabel("模型未加载")
        self.model_status_label.setAlignment(Qt.AlignCenter)
        self.model_status_label.setStyleSheet("color: #FF5252; font-size: 12px;")
        self.is_seeking = False
        self.save_dir = os.path.join(os.getcwd(), "saved_frames")  # 添加保存目录

        # 按钮高亮状态
        self.active_button = None
        self.button_highlight_timer = QTimer(self)
        self.button_highlight_timer.timeout.connect(self.clear_button_highlight)
        self.button_highlight_duration = 300  # 高亮持续时间(毫秒)

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

        main_layout.addLayout(top_layout)

        # 视频显示区域
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background-color: black;")
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

        # 进度条
        self.progress_bar = QSlider(Qt.Horizontal)
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
        main_layout.addLayout(info_layout)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # 快捷键提示
        self.shortcut_label = QLabel(
            "快捷键: ←/→=快退/快进 | 空格=播放/暂停 | ↑/↓=切换视频 | A/D=单帧进退 | 推理后可生成统计图表 | 自动保存: 已溢→images文件夹, 连续高置信度→problem文件夹")
        self.shortcut_label.setAlignment(Qt.AlignCenter)
        self.shortcut_label.setStyleSheet("font-weight: bold; color: #555; padding: 5px;")
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

            # 扫描支持的视频文件
            extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
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

        # 初始化视频捕获
        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", f"无法打开视频文件: {video_file}")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_pos = 0
        self.playing = False
        self.play_btn.setText("播放")

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


        # 显示第一帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.original_frame = frame.copy()  # 保存原始帧
            self.display_frame(frame)
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
            
        self.current_frame_pos = position
        
        if not self.playing:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_pos)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.original_frame = frame.copy()  # 更新原始帧
                # 如果处于推理状态，运行推理
                if self.inference_active:
                    self.run_inference(frame)
                self.display_frame(frame)
                self.frame_info_label.setText(f"当前帧：{self.current_frame_pos + 1} / 总帧数：{self.total_frames}")
        
        self.progress_bar.setValue(self.current_frame_pos)
        self.setFocus()

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
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    self.current_frame = frame
                    self.original_frame = frame.copy()  # 更新原始帧
                    self.progress_bar.setValue(self.current_frame_pos)
                    
                    # 如果处于推理状态，运行推理
                    if self.inference_active:
                        self.run_inference(frame)
                        
                    self.display_frame(frame)
                    self.frame_info_label.setText(f"当前帧：{self.current_frame_pos + 1} / 总帧数：{self.total_frames}")
                else:
                    # 视频结束，重置到第一帧
                    self.current_frame_pos = 0
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.toggle_play()
                    
                    # 如果处于推理状态，自动生成统计图表
                    if self.inference_active and self.frame_statistics:
                        self.generate_statistics_chart()
                        self.status_bar.showMessage("视频播放完成，统计图表已自动生成")

    def display_frame(self, frame):
        if frame is None:
            return

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
            Qt.Key_Down: None
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
            self.detection_model = YOLO(file_path)
            self.status_bar.showMessage(f"检测模型已加载: {os.path.basename(file_path)}")
            self.highlight_button(self.load_model_btn)
            
            # 检查模型状态
            self.check_model_status()
        except Exception as e:
            QMessageBox.warning(self, "模型加载错误", f"无法加载检测模型: {str(e)}")
            self.model_loaded = False

    def manage_classification_models(self):
        """打开分类模型管理对话框"""
        dialog = ModelManagerDialog(self)
        dialog.set_models(self.classification_models)
        dialog.exec_()
    
    def add_classification_model(self, model, name):
        """添加一个新的分类模型"""
        self.classification_models.append({
            'model': model,
            'name': name
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
        detection_path = 'pot_424.pt'
        
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

        radius_inner = int(radius * 0.36)
        radius_outer = int(radius * 0.08)

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



                    # 保存圆环图像
                    # 获取当前视频名（无扩展名）
                    video_file = os.path.join(self.video_folder, self.video_files[self.current_video_index])
                    video_name = os.path.splitext(os.path.basename(video_file))[0]

                    # 构建保存目录：ring/视频名/
                    ring_dir = os.path.join("ring", video_name)
                    os.makedirs(ring_dir, exist_ok=True)

                    # 构建文件名：frame_帧编号_crop_第几个目标.jpg
                    ring_filename = f"frame_{self.current_frame_pos}.jpg"
                    ring_path = os.path.join(ring_dir, ring_filename)

                    # 保存图像
                    cv2.imwrite(ring_path, image)

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
                                'class_index': predicted_class_index,
                                'class_name': predicted_class,
                                'confidence': confidence
                            })

                    results.append({
                        'box': (x1_new, y1_new, x2_new, y2_new),
                        'model_results': model_results
                    })

            self.inference_results = results

            # 统计当前帧的检测结果
            self.update_frame_statistics(self.current_frame_pos, results)

            # === 画框并保存整帧图像 ===
            output_frame = frame.copy()
            for item in results:
                (x1, y1, x2, y2) = item['box']

                # 只取第一个模型的分类结果
                model_result = item['model_results'][0]
                predicted_class = model_result['class_name']
                confidence = model_result['confidence']
                label = f"{predicted_class} {confidence:.2f}"

                # 设置颜色和线宽
                if predicted_class == 'boiling over':
                    color = (0, 0, 255)  # 红色 (BGR)
                    thickness = 10
                else:
                    color = (0, 255, 0)  # 绿色
                    thickness = 10

                # 画框和标签
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(output_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 获取当前视频名（无扩展名）
            video_file = os.path.join(self.video_folder, self.video_files[self.current_video_index])
            video_name = os.path.splitext(os.path.basename(video_file))[0]

            # 构建保存目录：result/视频名/
            frame_save_dir = os.path.join("result", video_name)
            os.makedirs(frame_save_dir, exist_ok=True)

            # 保存图像，命名格式与圆环图保持一致
            save_path = os.path.join(frame_save_dir, f"frame_{self.current_frame_pos}.jpg")
            cv2.imwrite(save_path, output_frame)

            return results

        except Exception as e:
            print(f"推理过程中出错: {str(e)}")
            self.inference_results = None
            return None

    def update_frame_statistics(self, frame_num, results):
        """更新当前帧的统计数据"""
        if results is None:
            return
            
        # 初始化当前帧统计
        frame_stats = {0: 0, 1: 0}
        current_frame_confidence = None
        current_frame_class = None
        
        # 统计每个检测结果
        for result in results:
            for model_result in result['model_results']:
                class_index = model_result['class_index']
                confidence = model_result['confidence']
                if class_index in [0, 1]:
                    frame_stats[class_index] += 1
                    # 记录当前帧的最高置信度和对应类别
                    if current_frame_confidence is None or confidence > current_frame_confidence:
                        current_frame_confidence = confidence
                        current_frame_class = class_index
        
        # 保存帧统计
        self.frame_statistics[frame_num] = frame_stats
        
        # 更新视频总统计
        self.video_statistics[0] += frame_stats[0]
        self.video_statistics[1] += frame_stats[1]
        
        # 自动保存逻辑
        self.auto_save_frames(frame_num, current_frame_class, current_frame_confidence)
        
        # 更新上一帧信息
        self.previous_frame_confidence = current_frame_confidence
        self.previous_frame_class = current_frame_class
        
        # 如果有统计数据，启用图表生成按钮
        if self.video_statistics[0] > 0 or self.video_statistics[1] > 0:
            self.generate_chart_btn.setEnabled(True)
    
    def create_annotated_frame(self, frame):
        """创建带有检测框和分类信息的标注图片"""
        if self.inference_results is None or frame is None:
            return frame.copy()
            
        # 复制原始帧用于标注
        annotated_frame = frame.copy()
        
        # 绘制检测结果
        for result in self.inference_results:
            box = result['box']
            x1, y1, x2, y2 = box
            
            # 为每个模型的结果绘制边界框和标签
            for i, model_result in enumerate(result['model_results']):
                class_name = model_result['class_name']
                confidence = model_result['confidence']
                model_name = model_result['model_name']
                color = class_colors.get(class_name, (255, 255, 255))
                
                # 只绘制一次边界框
                if i == 0:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                
                # 绘制模型名称和结果
                text = f"{model_name[:10]}: {class_name} ({confidence:.3f})"
                
                # 计算文本背景框的大小
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                
                # 绘制文本背景
                text_y = y1 + 35 * i + 25
                cv2.rectangle(annotated_frame, 
                             (x1, text_y - text_height - 5), 
                             (x1 + text_width + 10, text_y + 5), 
                             color, -1)
                
                # 绘制文本（使用黑色以确保可见性）
                cv2.putText(annotated_frame, text, (x1 + 5, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return annotated_frame
    
    def auto_save_frames(self, frame_num, current_class, current_confidence):
        """自动保存帧图片的逻辑"""
        if self.original_frame is None or current_class is None:
            return
            
        try:
            # 获取当前视频名称（不含扩展名）
            video_name = "unknown"
            if self.current_video_index >= 0 and self.current_video_index < len(self.video_files):
                video_name = os.path.splitext(self.video_files[self.current_video_index])[0]
            
            # 条件1: 如果检测到已溢（类别1），保存到images文件夹下的视频名称子文件夹
            if current_class == 1:
                # 创建images文件夹下的视频名称子文件夹
                video_images_dir = os.path.join(self.images_save_dir, video_name)
                os.makedirs(video_images_dir, exist_ok=True)
                
                # 生成文件名（不再包含视频名称，因为已经在文件夹名中）
                filename = f"frame_{frame_num}_boiling_over.jpg"
                filepath = os.path.join(video_images_dir, filename)
                
                # 创建带有检测框和分类信息的标注图片
                annotated_frame = self.create_annotated_frame(self.original_frame)
                
                # 保存标注图片
                cv2.imwrite(filepath, annotated_frame)
                print(f"已溢帧已保存（带检测框）: {filepath}")
                self.status_bar.showMessage(f"已溢帧已自动保存到 {video_name} 文件夹: {filename}")
                
                # 条件2: 如果检测到已溢（类别1）且连续两帧置信度都超过0.9，保存到problem文件夹
                if (current_confidence is not None and current_confidence > 0.9 and 
                    self.previous_frame_confidence is not None and self.previous_frame_confidence > 0.9):
                    
                    # 创建problem文件夹下的视频名称子文件夹
                    video_problem_dir = os.path.join(self.problem_save_dir, video_name)
                    os.makedirs(video_problem_dir, exist_ok=True)
                    
                    # 生成文件名（不再包含视频名称，因为已经在文件夹名中）
                    filename = f"frame_{frame_num}_problem.jpg"
                    filepath = os.path.join(video_problem_dir, filename)
                    
                    # 创建带有检测框和分类信息的标注图片
                    annotated_frame = self.create_annotated_frame(self.original_frame)
                    
                    # 保存标注图片
                    cv2.imwrite(filepath, annotated_frame)
                    print(f"问题帧已保存（已溢且高置信度）: {filepath}")
                    self.status_bar.showMessage(f"问题帧已自动保存到 {video_name} 文件夹: {filename}")
                
        except Exception as e:
            print(f"自动保存帧时出错: {str(e)}")
    
    def generate_statistics_chart(self):
        """生成并保存统计图表"""
        if not self.frame_statistics or self.current_video_index < 0:
            QMessageBox.warning(self, "无统计数据", "请先进行视频推理以生成统计数据")
            return
            
        try:
            # 确保chart目录存在
            os.makedirs(self.chart_save_dir, exist_ok=True)
            
            # 获取当前视频名称（不含扩展名）
            video_name = os.path.splitext(self.video_files[self.current_video_index])[0]
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 子图1：总体统计饼图
            labels = ['未溢出 (0)', '已溢出 (1)']
            sizes = [self.video_statistics[0], self.video_statistics[1]]
            colors = ['#66b3ff', '#ff9999']
            
            # 只显示有数据的部分
            non_zero_labels = []
            non_zero_sizes = []
            non_zero_colors = []
            for i, size in enumerate(sizes):
                if size > 0:
                    non_zero_labels.append(f"{labels[i]}\n({size}次)")
                    non_zero_sizes.append(size)
                    non_zero_colors.append(colors[i])
            
            if non_zero_sizes:
                ax1.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors, 
                       autopct='%1.1f%%', startangle=90)
                ax1.set_title(f'视频总体检测统计\n总检测次数: {sum(sizes)}', fontsize=14, fontweight='bold')
            else:
                ax1.text(0.5, 0.5, '无检测数据', ha='center', va='center', transform=ax1.transAxes, fontsize=16)
                ax1.set_title('视频总体检测统计', fontsize=14, fontweight='bold')
            
            # 子图2：帧级别统计柱状图
            frame_nums = sorted(self.frame_statistics.keys())
            class_0_counts = [self.frame_statistics[f][0] for f in frame_nums]
            class_1_counts = [self.frame_statistics[f][1] for f in frame_nums]
            
            x = np.arange(len(frame_nums))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, class_0_counts, width, label='未溢出 (0)', color='#66b3ff', alpha=0.8)
            bars2 = ax2.bar(x + width/2, class_1_counts, width, label='已溢出 (1)', color='#ff9999', alpha=0.8)
            
            ax2.set_xlabel('帧编号', fontsize=12)
            ax2.set_ylabel('检测次数', fontsize=12)
            ax2.set_title('各帧检测结果统计', fontsize=14, fontweight='bold')
            ax2.set_xticks(x[::max(1, len(frame_nums)//10)])  # 最多显示10个刻度
            ax2.set_xticklabels([str(frame_nums[i]) for i in range(0, len(frame_nums), max(1, len(frame_nums)//10))])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 在柱状图上添加数值标签
            def add_value_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{int(height)}', ha='center', va='bottom', fontsize=8)
            
            add_value_labels(bars1)
            add_value_labels(bars2)
            
            plt.tight_layout()
            
            # 保存图表
            chart_filename = os.path.join(self.chart_save_dir, f"{video_name}.png")
            plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 显示成功消息
            total_detections = self.video_statistics[0] + self.video_statistics[1]
            message = f"统计图表已保存！\n\n" \
                     f"文件路径: {chart_filename}\n\n" \
                     f"统计结果:\n" \
                     f"• 未溢出 (0): {self.video_statistics[0]} 次\n" \
                     f"• 已溢出 (1): {self.video_statistics[1]} 次\n" \
                     f"• 总检测次数: {total_detections} 次\n" \
                     f"• 统计帧数: {len(self.frame_statistics)} 帧"
            
            QMessageBox.information(self, "图表生成成功", message)
            self.status_bar.showMessage(f"统计图表已保存: {chart_filename}")
            self.highlight_button(self.generate_chart_btn)
            
        except Exception as e:
            QMessageBox.critical(self, "图表生成失败", f"生成统计图表时发生错误:\n{str(e)}")
            print(f"图表生成错误: {str(e)}")

    def draw_inference_results(self, frame):
        """在帧上绘制推理结果"""
        if self.inference_results is None:
            return frame
            
        for result in self.inference_results:
            box = result['box']
            x1, y1, x2, y2 = box
            
            # 为每个模型的结果绘制边界框和标签
            for i, model_result in enumerate(result['model_results']):
                class_name = model_result['class_name']
                confidence = model_result['confidence']
                model_name = model_result['model_name']
                color = class_colors.get(class_name, (255, 255, 255))
                
                # 只绘制一次边界框
                if i == 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制模型名称和结果
                text = f"{model_name[:10]}: {class_name} ({confidence:.2f})"
                cv2.putText(frame, text, (x1, y1 + 30 * i + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame

    # 按钮高亮方法
    def highlight_button(self, button):
        """高亮显示指定的按钮"""
        # 清除之前的高亮
        self.clear_button_highlight()
        
        # 设置新的高亮按钮
        self.active_button = button
        self.update_button_styles()
        
        # 设置定时器清除高亮
        self.button_highlight_timer.start(self.button_highlight_duration)

    def clear_button_highlight(self):
        """清除按钮高亮状态"""
        self.active_button = None
        self.button_highlight_timer.stop()
        self.update_button_styles()

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()


if __name__ == "__main__":
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