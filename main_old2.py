import sys
import cv2
import numpy as np
import requests
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QLabel, QFileDialog, QWidget, QTabWidget, QScrollArea, QGridLayout,
                             QLineEdit, QMessageBox, QRadioButton, QComboBox)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QFontDatabase
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
import pyttsx3
import threading
import torch
from sort import Sort
import math
from collections import deque
import os
from datetime import datetime
import sqlite3
import easyocr
import queue

def process_frame(frame):
    lower_range = np.array([58, 97, 222])
    upper_range = np.array([179, 255, 255])
    lower_range1 = np.array([0, 43, 184])
    upper_range1 = np.array([56, 132, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask1 = cv2.inRange(hsv, lower_range1, upper_range1)
    combined_mask = cv2.bitwise_or(mask, mask1)
    _, final_mask = cv2.threshold(combined_mask, 254, 255, cv2.THRESH_BINARY)
    detected_label = None

    cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        if cv2.contourArea(c) > 50:
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2
            cy = y + h // 2
            if cx < 915:
                if cv2.countNonZero(mask[y:y+h, x:x+w]) > 0:
                    color = (0, 255, 0)
                    text_color = (0, 255, 0)
                    label = "GREEN"
                elif cv2.countNonZero(mask1[y:y+h, x:x+w]) > 0:
                    color = (0, 0, 255)
                    text_color = (0, 0, 255)
                    label = "RED"
                else:
                    continue
                detected_label = label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.circle(frame, (cx, cy), 1, (255, 0, 0), -1)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    return frame, detected_label

class VideoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Дорожное приложение")
        self.setMinimumSize(1366, 768)

        QFontDatabase().addApplicationFont("fonts/Inter-Regular.ttf")
        QFontDatabase().addApplicationFont("fonts/Inter-Bold.ttf")

        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E2E;
                font-family: 'Inter', sans-serif;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6366F1, stop:1 #818CF8);
                color: white;
                border-radius: 12px;
                padding: 8px 12px;
                font-size: 13px;
                font-weight: bold;
                border: none;
                text-align: center;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #818CF8, stop:1 #A5B4FC);
            }
            QPushButton:pressed {
                background: #4F46E5;
            }
            QLineEdit {
                background-color: #2A2A3E;
                color: #E5E7EB;
                border: 1px solid #3F3F5A;
                border-radius: 8px;
                padding: 8px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #6366F1;
            }
            QComboBox {
                background-color: #2A2A3E;
                color: #E5E7EB;
                border: 1px solid #3F3F5A;
                border-radius: 8px;
                padding: 8px;
                font-size: 13px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2A2A3E;
                color: #E5E7EB;
                selection-background-color: #6366F1;
            }
            QLabel#videoLabel {
                background-color: #2A2A3E;
                border: 3px solid #3F3F5A;
                border-radius: 12px;
            }
            QLabel#statusLabel {
                background-color: #2A2A3E;
                color: #E5E7EB;
                font-size: 14px;
                padding: 10px;
                border-radius: 8px;
            }
            QLabel#trafficLightIndicator {
                background-color: rgba(63, 63, 90, 0.9);
                color: #E5E7EB;
                font-size: 14px;
                padding: 10px;
                border-radius: 12px;
                border: 1px solid #6366F1;
            }
            QTabWidget::pane {
                border: 2px solid #3F3F5A;
                border-radius: 12px;
                background-color: #2A2A3E;
            }
            QTabWidget::tab-bar {
                alignment: center;
            }
            QTabBar::tab {
                background-color: #3F3F5A;
                color: #E5E7EB;
                padding: 12px 24px;
                border-radius: 8px;
                margin: 4px;
                font-size: 16px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6366F1, stop:1 #818CF8);
                color: white;
            }
            QWidget#sidebar {
                background-color: #2A2A3E;
                border-right: 2px solid #3F3F5A;
            }
            QWidget#violationCard {
                background-color: #3F3F5A;
                border-radius: 12px;
                padding: 10px;
                margin: 5px;
            }
            QWidget#violationCard:hover {
                background-color: #4B5563;
            }
            QLabel#violationLabel {
                color: #E5E7EB;
                font-size: 14px;
            }
            QScrollArea {
                background-color: #2A2A3E;
                border: none;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Используемое устройство: {self.device}")
        self.model_signs = YOLO("best.pt").to(self.device)
        self.model_vehicles = YOLO("yolov8s.pt").to(self.device)
        self.model_crosswalk = YOLO("yaya.pt").to(self.device)

        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if "turkish" in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break

        self.speech_queue = queue.Queue()
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()

        self.spoken_labels = set()
        self.tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.2)
        self.fps = 30
        self.meters_per_pixel_base = 0.07
        self.horizon_y = 50
        self.perspective_factor = 1000
        self.positions = {}
        self.speed_history = {}
        self.trajectories = {}
        self.max_speed_history = 10
        self.min_display_speed = 5
        self.max_speed_limit = 150
        self.speed_limit = 50
        self.google_speed_limit = None
        self.use_google_speed = True
        self.max_pixel_displacement = 50
        self.area = [(324, 313), (283, 374), (854, 392), (864, 322)]
        self.violated_ids = []
        self.lane_violated_ids = []
        self.pedestrian_violated_ids = []
        self.red_light_violated_ids = []
        self.pedestrian_stop_counters = {}
        self.previous_lanes = {}
        self.lane_change_counters = {}
        self.min_frames_for_lane_change = 10
        self.lane_change_threshold = 0.7
        self.lane_tolerance = 10
        self.today_date = datetime.now().strftime('%Y-%m-%d')
        self.output_dir = os.path.join('saved_images', self.today_date)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.traffic_light_mode = False
        self.pedestrian_area = None
        self.pedestrian_check_interval = 15
        self.was_inside_crosswalk = {}

        # Координаты линии для нарушения красного света (по умолчанию)
        self.traffic_line = [(300, 400), (720, 400)]  # Координаты линии: [(x1, y1), (x2, y2)]

        self.init_db()
        self.ocr_reader = easyocr.Reader(['en'])

        self.google_api_key = "AIzaSyAFOjF4yHGaDt0PixynVjSWMc1ktjPkVwo"

        main_layout = QHBoxLayout()
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(280)
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setAlignment(Qt.AlignTop)
        sidebar_layout.setSpacing(10)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)

        logo_label = QLabel("Дорожное приложение")
        logo_label.setStyleSheet("color: #E5E7EB; font-size: 20px; font-weight: bold;")
        slogan_label = QLabel("Для безопасных дорог")
        slogan_label.setStyleSheet("color: #A5B4FC; font-size: 12px;")
        sidebar_layout.addWidget(logo_label)
        sidebar_layout.addWidget(slogan_label)
        sidebar_layout.addSpacing(20)

        self.btn_camera = QPushButton("Камера компьютера")
        self.btn_camera.setIcon(QIcon("icons/camera.svg"))
        self.btn_phone_camera = QPushButton("Камера телефона")
        self.btn_phone_camera.setIcon(QIcon("icons/phone.svg"))
        self.btn_video = QPushButton("Видео")
        self.btn_video.setIcon(QIcon("icons/video.svg"))
        self.btn_traffic_light = QPushButton("Управление светофором")
        self.btn_traffic_light.setIcon(QIcon("icons/traffic-light.svg"))
        self.btn_stop = QPushButton("Остановить")
        self.btn_stop.setIcon(QIcon("icons/stop.svg"))
        self.btn_exit = QPushButton("Выход")
        self.btn_exit.setIcon(QIcon("icons/exit.svg"))

        for btn in [self.btn_camera, self.btn_phone_camera, self.btn_video, self.btn_traffic_light, self.btn_stop, self.btn_exit]:
            btn.setFixedHeight(50)
            sidebar_layout.addWidget(btn)

        speed_limit_label = QLabel("Ограничение скорости (км/ч):")
        speed_limit_label.setStyleSheet("color: #E5E7EB; font-size: 14px;")
        self.use_google_speed_limit = QRadioButton("Использовать лимит Google Maps")
        self.use_manual_speed_limit = QRadioButton("Ввести лимит вручную")
        self.use_google_speed_limit.setChecked(True)
        self.speed_limit_input = QLineEdit()
        self.speed_limit_input.setPlaceholderText("Введите ограничение скорости (например, 50)")
        self.speed_limit_input.setText(str(self.speed_limit))
        self.speed_limit_input.setEnabled(False)
        self.btn_set_speed_limit = QPushButton("Установить ограничение скорости")
        self.btn_set_speed_limit.setFixedHeight(40)

        self.use_google_speed_limit.toggled.connect(self.toggle_speed_limit_input)
        self.use_manual_speed_limit.toggled.connect(self.toggle_speed_limit_input)

        sidebar_layout.addSpacing(20)
        sidebar_layout.addWidget(speed_limit_label)
        sidebar_layout.addWidget(self.use_google_speed_limit)
        sidebar_layout.addWidget(self.use_manual_speed_limit)
        sidebar_layout.addWidget(self.speed_limit_input)
        sidebar_layout.addWidget(self.btn_set_speed_limit)

        sidebar.setLayout(sidebar_layout)
        main_layout.addWidget(sidebar)

        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(20, 20, 20, 20)

        self.status_label = QLabel(f"Устройство: {self.device} | FPS: 0 | Режим: Стандартный | Ограничение скорости: {self.speed_limit} км/ч")
        self.status_label.setObjectName("statusLabel")
        content_layout.addWidget(self.status_label)

        self.tab_widget = QTabWidget()
        self.video_tab = QWidget()
        self.violations_tab = QWidget()
        self.report_tab = QWidget()
        self.camera_selection_tab = QWidget()

        self.tab_widget.addTab(self.video_tab, "Видеопоток")
        self.tab_widget.addTab(self.violations_tab, "Нарушения красного света")
        self.tab_widget.addTab(self.report_tab, "Отчет о нарушениях")
        self.tab_widget.addTab(self.camera_selection_tab, "Выбор камеры")

        video_layout = QVBoxLayout()
        self.video_container = QWidget()
        self.video_container.setStyleSheet("background-color: #2A2A3E; border-radius: 12px; position: relative;")
        self.image_label = QLabel()
        self.image_label.setObjectName("videoLabel")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.traffic_light_indicator = QLabel("Светофор: Неизвестно")
        self.traffic_light_indicator.setObjectName("trafficLightIndicator")
        self.traffic_light_indicator.setFixedSize(200, 40)
        self.traffic_light_indicator.setParent(self.video_container)
        self.traffic_light_indicator.move(10, 10)

        video_container_layout = QVBoxLayout()
        video_container_layout.addWidget(self.image_label)
        self.video_container.setLayout(video_container_layout)
        video_layout.addWidget(self.video_container, stretch=1)

        self.video_tab.setLayout(video_layout)

        violations_layout = QVBoxLayout()
        self.violations_scroll = QScrollArea()
        self.violations_scroll.setWidgetResizable(True)
        self.violations_container = QWidget()
        self.violations_layout = QVBoxLayout()
        self.violations_layout.setAlignment(Qt.AlignTop)
        self.violations_container.setLayout(self.violations_layout)
        self.violations_scroll.setWidget(self.violations_container)
        violations_layout.addWidget(self.violations_scroll)
        self.violations_tab.setLayout(violations_layout)

        report_layout = QVBoxLayout()
        self.report_label = QLabel("Отчет о нарушениях будет показан здесь.")
        report_layout.addWidget(self.report_label)
        self.report_tab.setLayout(report_layout)

        camera_selection_layout = QVBoxLayout()
        camera_selection_layout_inner = QHBoxLayout()
        camera_label = QLabel("Выберите камеру:")
        camera_label.setStyleSheet("color: #E5E7EB; font-size: 14px;")
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Камера 1", "Камера 2"])
        camera_selection_layout_inner.addWidget(camera_label)
        camera_selection_layout_inner.addWidget(self.camera_combo)
        camera_selection_layout.addLayout(camera_selection_layout_inner)

        self.camera_video_label = QLabel()
        self.camera_video_label.setObjectName("videoLabel")
        self.camera_video_label.setAlignment(Qt.AlignCenter)
        camera_selection_layout.addWidget(self.camera_video_label)

        self.camera_selection_tab.setLayout(camera_selection_layout)

        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        self.camera_combo.currentTextChanged.connect(self.load_camera_video)

        content_layout.addWidget(self.tab_widget)
        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget, stretch=1)

        central_widget.setLayout(main_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frame)

        self.cap = None
        self.camera_cap = None
        self.detected_labels = set()
        self.is_camera = False
        self.frame_id = 0
        self.last_fps_time = datetime.now()

        self.video_paths = {
            "Камера 1": "video1.mp4",
            "Камера 2": "video2.mp4"
        }

        self.btn_camera.clicked.connect(self.start_camera)
        self.btn_phone_camera.clicked.connect(self.start_phone_camera)
        self.btn_video.clicked.connect(self.open_video)
        self.btn_traffic_light.clicked.connect(self.toggle_traffic_light_mode)
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_exit.clicked.connect(self.close)
        self.btn_set_speed_limit.clicked.connect(self.set_speed_limit)

        # Координаты пешеходного перехода для video1
        self.video1_pedestrian_area = [
            (350, 290),  # Левый нижний
            (190, 365),  # Левый верхний
            (950, 490),  # Правый верхний
            (970, 370)   # Правый нижний
        ]

    def toggle_speed_limit_input(self):
        self.use_google_speed = self.use_google_speed_limit.isChecked()
        self.speed_limit_input.setEnabled(not self.use_google_speed)
        if self.use_google_speed:
            self.speed_limit_input.setText("Используется ограничение скорости Google Maps")
            self.fetch_google_speed_limit()
        else:
            self.speed_limit_input.setText(str(self.speed_limit))

    def fetch_google_speed_limit(self):
        latitude, longitude = 41.0082, 28.9784
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
        way(around:10,{latitude},{longitude})[highway][maxspeed];
        out body;
        """
        try:
            response = requests.post(overpass_url, data=overpass_query)
            if response.status_code == 200:
                data = response.json()
                if data["elements"]:
                    speed_limit = data["elements"][0]["tags"].get("maxspeed", "50")
                    self.google_speed_limit = float(speed_limit) if speed_limit.isdigit() else 50
                    print(f"Ограничение скорости OSM: {self.google_speed_limit} км/ч")
                else:
                    print("Ограничение скорости не найдено: Ответ не содержит данных.")
                    self.google_speed_limit = 50
            else:
                print(f"Ошибка API OSM: HTTP {response.status_code}")
                self.google_speed_limit = 50
        except Exception as e:
            print(f"Ошибка API OSM: {str(e)}")
            self.google_speed_limit = 50
        self.update_status()

    def set_speed_limit(self):
        if not self.use_google_speed:
            try:
                new_speed_limit = int(self.speed_limit_input.text())
                if new_speed_limit <= 0:
                    QMessageBox.warning(self, "Ошибка", "Ограничение скорости должно быть положительным числом!")
                    return
                self.speed_limit = new_speed_limit
                self.update_status()
                QMessageBox.information(self, "Успешно", f"Ограничение скорости установлено на {self.speed_limit} км/ч.")
            except ValueError:
                QMessageBox.warning(self, "Ошибка", "Пожалуйста, введите корректное число!")
        else:
            QMessageBox.information(self, "Информация", "Используется ограничение скорости Google Maps. Ручной ввод отключен.")

    def on_tab_changed(self, index):
        if self.tab_widget.tabText(index) == "Выбор камеры":
            self.camera_combo.setCurrentText("Камера 1")
            self.load_camera_video("Камера 1")
        else:
            self.camera_timer.stop()
            if self.camera_cap:
                self.camera_cap.release()
                self.camera_cap = None
            self.camera_video_label.clear()

    def load_camera_video(self, camera_name):
        self.camera_timer.stop()
        if self.camera_cap:
            self.camera_cap.release()
            self.camera_cap = None

        video_path = self.video_paths.get(camera_name)
        if video_path:
            self.camera_cap = cv2.VideoCapture(video_path)
            if not self.camera_cap.isOpened():
                QMessageBox.critical(self, "Ошибка", f"Не удалось открыть файл {video_path}!")
                self.camera_cap = None
                return
            fps = self.camera_cap.get(cv2.CAP_PROP_FPS)
            self.fps = int(fps) if fps > 0 else 30
            print(f"FPS видео: {self.fps}")

            if camera_name == "Камера 1":
                self.camera_timer.timeout.disconnect()
                self.camera_timer.timeout.connect(self.update_camera_frame_video1)
            else:
                self.camera_timer.timeout.disconnect()
                self.camera_timer.timeout.connect(self.update_camera_frame)

            self.camera_timer.start(int(1000 // max(1, self.fps)))

    def update_camera_frame(self):
        if self.camera_cap and self.camera_cap.isOpened():
            ret, frame = self.camera_cap.read()
            if not ret:
                self.camera_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return

            self.frame_id += 1
            original_h, original_w = frame.shape[:2]

            self.camera_video_label.setFixedSize(self.video_container.size())
            self.camera_video_label.setScaledContents(True)

            annotated_frame = frame.copy()

            if self.camera_combo.currentText() == "Камера 2" and "speed_zone" not in self.__dict__:
                self.speed_zone = None
            if self.speed_zone is None:
                display_width, display_height = original_w, original_h
                self.speed_zone = {
                    'x1': int(display_width * 0.2),
                    'y1': int(display_height * 0.4),
                    'x2': int(display_width * 0.8),
                    'y2': int(display_height * 0.7)
                }
                zone_pixel_length = self.speed_zone['y2'] - self.speed_zone['y1']
                self.real_distance_meters = 10.0
                self.meters_per_pixel = self.real_distance_meters / zone_pixel_length
                print(f"Зона измерения скорости определена: {self.speed_zone}, метров на пиксель: {self.meters_per_pixel}")

            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (self.speed_zone['x1'], self.speed_zone['y1']),
                          (self.speed_zone['x2'], self.speed_zone['y2']),
                          (0, 255, 255), -1)
            alpha = 0.1
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

            cv2.rectangle(annotated_frame, (self.speed_zone['x1'], self.speed_zone['y1']),
                          (self.speed_zone['x2'], self.speed_zone['y2']),
                          (0, 255, 255), 1)

            results_signs = self.model_signs.predict(
                source=frame,
                conf=0.5,
                imgsz=max(original_w, original_h),
                show=False,
                verbose=False,
                device=self.device
            )
            boxes_signs = results_signs[0].boxes
            labels_on_frame = set()

            for box in boxes_signs:
                cls_id = int(box.cls[0])
                label = self.model_signs.names[cls_id]
                cleaned_label = self.clean_label(label)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = tuple(int(x) for x in np.random.default_rng(abs(hash(cleaned_label)) % (2**32)).integers(0, 255, size=3))

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                text = f"{cleaned_label}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated_frame, text, (x1 + 2, y1 - 7),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                labels_on_frame.add(cleaned_label)

                if cleaned_label not in self.spoken_labels:
                    self.spoken_labels.add(cleaned_label)
                    self.speak_label(cleaned_label)

            y_offset = 40
            for label in sorted(labels_on_frame):
                color = tuple(int(x) for x in np.random.default_rng(abs(hash(label)) % (2**32)).integers(0, 255, size=3))
                cv2.putText(annotated_frame, label,
                           (15, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                y_offset += 40

            results_vehicles = self.model_vehicles.predict(
                source=frame,
                conf=0.25,
                imgsz=max(original_w, original_h),
                show=False,
                verbose=False,
                device=self.device
            )
            detections = []

            for r in results_vehicles:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    if cls in [2, 3, 5, 7]:
                        detections.append([x1, y1, x2, y2])

            detections = np.array(detections)
            if len(detections) > 0:
                tracked_objects = self.tracker.update(detections)
                print(f"Обнаружено транспортных средств: {len(tracked_objects)}")
            else:
                tracked_objects = []
                print("Транспортные средства не обнаружены.")

            for obj in tracked_objects:
                x1, y1, x2, y2, track_id = obj
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                track_id = int(track_id)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if track_id not in self.trajectories:
                    self.trajectories[track_id] = deque(maxlen=30)
                self.trajectories[track_id].append((cx, cy))

                in_zone = cv2.pointPolygonTest(np.array([(self.speed_zone['x1'], self.speed_zone['y1']),
                                                         (self.speed_zone['x1'], self.speed_zone['y2']),
                                                         (self.speed_zone['x2'], self.speed_zone['y2']),
                                                         (self.speed_zone['x2'], self.speed_zone['y1'])], np.int32),
                                               (cx, cy), False) >= 0
                print(f"Автомобиль ID {track_id} в зоне измерения скорости? {in_zone}, Координаты: ({cx}, {cy})")

                if in_zone and track_id in self.positions:
                    prev_cx, prev_cy, prev_frame = self.positions[track_id]
                    distance_pixels = abs(cy - prev_cy)
                    print(f"Для автомобиля ID {track_id} distance_pixels: {distance_pixels}")

                    if distance_pixels < 2:
                        distance_pixels = 0
                        print(f"Автомобиль ID {track_id}: distance_pixels слишком мал, обнулен.")
                    elif distance_pixels > self.max_pixel_displacement:
                        distance_pixels = 0
                        print(f"Автомобиль ID {track_id}: distance_pixels слишком велик, обнулен.")

                    distance_meters = distance_pixels * self.meters_per_pixel
                    time_seconds = (self.frame_id - prev_frame) / max(1, self.fps)
                    print(f"Автомобиль ID {track_id}: distance_meters: {distance_meters}, time_seconds: {time_seconds}")

                    if time_seconds > 0 and distance_meters > 0:
                        speed_ms = distance_meters / time_seconds
                        speed_kmh = speed_ms * 3.6
                        if speed_kmh > self.max_speed_limit:
                            speed_kmh = self.max_speed_limit
                        print(f"Автомобиль ID {track_id}: Скорость рассчитана: {speed_kmh} км/ч")

                        if track_id not in self.speed_history:
                            self.speed_history[track_id] = deque(maxlen=self.max_speed_history)
                        if speed_kmh > 0:
                            self.speed_history[track_id].append(speed_kmh)

                        speed_violation, timestamp = self.detect_speed_violation(track_id, speed_kmh)
                        if speed_violation:
                            vehicle_img = frame[y1:y2, x1:x2]
                            vehicle_filename = f"Speed_ID_{track_id}_{timestamp.replace(' ', '_').replace(':', '-')}_vehicle.jpg"
                            vehicle_path = os.path.join(self.output_dir, vehicle_filename)
                            if vehicle_img.size > 0:
                                cv2.imwrite(vehicle_path, vehicle_img)
                            plate_y1 = y2 - int((y2 - y1) * 0.3)
                            plate_img = frame[plate_y1:y2, x1:x2]
                            plate_filename = f"Speed_ID_{track_id}_{timestamp.replace(' ', '_').replace(':', '-')}_plate.jpg"
                            plate_path = os.path.join(self.output_dir, plate_filename)
                            if plate_img.size > 0:
                                cv2.imwrite(plate_path, plate_img)
                            self.add_violation_card(track_id, timestamp, "Нарушение скоростного режима", vehicle_path, plate_path)

                    else:
                        print(f"Автомобиль ID {track_id}: Скорость не рассчитана (time_seconds: {time_seconds}, distance_meters: {distance_meters})")

                self.positions[track_id] = (cx, cy, self.frame_id)

                hue = (track_id * 15) % 180
                color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                color = (int(color[0]), int(color[1]), int(color[2]))

                for i in range(1, len(self.trajectories[track_id])):
                    cv2.line(annotated_frame, self.trajectories[track_id][i-1],
                             self.trajectories[track_id][i], color, 2)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(annotated_frame, f'ID:{track_id}', (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if in_zone and track_id in self.speed_history and len(self.speed_history[track_id]) > 0:
                    avg_speed = int(sum(self.speed_history[track_id]) / len(self.speed_history[track_id]))
                    print(f"Автомобиль ID {track_id}: Средняя скорость: {avg_speed} км/ч")
                    if avg_speed >= self.min_display_speed:
                        (text_width, text_height), _ = cv2.getTextSize(f"{avg_speed}km/h", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        bg_x1, bg_y1 = x1, y1 - text_height - 5
                        bg_x2, bg_y2 = x1 + text_width + 5, y1 + 5
                        cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
                        cv2.putText(annotated_frame, f"{avg_speed}km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.camera_video_label.setPixmap(pixmap)

            current_time = datetime.now()
            elapsed = (current_time - self.last_fps_time).total_seconds()
            if elapsed > 1:
                self.fps = int(self.frame_id / elapsed) if elapsed > 0 else 30
                self.frame_id = 0
                self.last_fps_time = current_time
                self.update_status()

    def update_camera_frame_video1(self):
        if self.camera_cap and self.camera_cap.isOpened():
            ret, frame = self.camera_cap.read()
            if not ret:
                self.camera_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return

            self.frame_id += 1
            frame = cv2.resize(frame, (1020, 600))
            annotated_frame = frame.copy()

            self.camera_video_label.setFixedSize(self.video_container.size())
            self.camera_video_label.setScaledContents(True)

            pedestrian_area = self.video1_pedestrian_area
            cv2.polylines(annotated_frame, [np.array(pedestrian_area, np.int32)], True, (255, 0, 255), 3)
            cv2.putText(annotated_frame, "pedestrian crossing", (pedestrian_area[0][0], pedestrian_area[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Video1 için trafik ışıkları (2 tane dörtgen, RED etiketiyle)
            traffic_light_1 = (190, 40, 40, 40)  # İlk trafik ışığı koordinatları (x, y, w, h)
            traffic_light_2 = (680, 135, 40, 40)  # İkinci trafik ışığı koordinatları (x, y, w, h)
            
            cv2.rectangle(annotated_frame, (traffic_light_1[0], traffic_light_1[1]), 
                          (traffic_light_1[0] + traffic_light_1[2], traffic_light_1[1] + traffic_light_1[3]), 
                          (0, 0, 255), 2)
            cv2.putText(annotated_frame, "RED", (traffic_light_1[0], traffic_light_1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(annotated_frame, (traffic_light_2[0], traffic_light_2[1]), 
                          (traffic_light_2[0] + traffic_light_2[2], traffic_light_2[1] + traffic_light_2[3]), 
                          (0, 0, 255), 2)
            cv2.putText(annotated_frame, "RED", (traffic_light_2[0], traffic_light_2[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            start_point = (100, 430)
            end_point = (450, 500)
            cv2.line(annotated_frame, start_point, end_point, (0, 0, 255), 2)  # Kırmızı çizgi

            # Yazı ekliyoruz (çizginin üstüne biraz yukarıda)
            text_position = (start_point[0], start_point[1] - 10)  # Yani (50, 190)
            cv2.putText(annotated_frame, "Traffic light", text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)




            results_vehicles = self.model_vehicles.predict(
                source=frame,
                conf=0.25,
                imgsz=640,
                show=False,
                verbose=False,
                device=self.device
            )
            for r in results_vehicles:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if cls == 0:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, "Person", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    elif cls in [2, 3, 5, 7]:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, "Car", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.camera_video_label.setPixmap(pixmap)

            current_time = datetime.now()
            elapsed = (current_time - self.last_fps_time).total_seconds()
            if elapsed > 1:
                self.fps = int(self.frame_id / elapsed) if elapsed > 0 else 30
                self.frame_id = 0
                self.last_fps_time = current_time
                self.update_status()

    def resizeEvent(self, event):
        window_size = self.centralWidget().size()
        self.video_container.setFixedSize(int(window_size.width() * 0.85), window_size.height() - 100)
        indicator_width = self.traffic_light_indicator.width()
        self.traffic_light_indicator.move(self.video_container.width() - indicator_width - 10, 10)
        super().resizeEvent(event)

    def start_camera(self):
        self.traffic_light_mode = False
        self.btn_traffic_light.setText("Управление светофором")
        self.stop_video()
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.is_camera = True
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = int(fps) if fps > 0 else 30
            print(f"FPS камеры: {self.fps}")
            self.timer.start(20)
        else:
            print("Не удалось открыть камеру компьютера!")

    def start_phone_camera(self):
        self.traffic_light_mode = False
        self.btn_traffic_light.setText("Управление светофором")
        self.stop_video()
        phone_camera_source = 1
        self.cap = cv2.VideoCapture(phone_camera_source)
        if self.cap.isOpened():
            self.is_camera = True
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = int(fps) if fps > 0 else 30
            print(f"FPS камеры телефона: {self.fps}")
            self.timer.start(20)
        else:
            print("Не удалось открыть камеру телефона! Проверьте настройки DroidCam/IP Webcam.")

    def open_video(self):
        self.traffic_light_mode = False
        self.btn_traffic_light.setText("Управление светофором")
        self.stop_video()
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите видео")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            if self.cap.isOpened():
                self.is_camera = False
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.fps = int(fps) if fps > 0 else 30
                print(f"FPS видео: {self.fps}")
                self.timer.start(25)
            else:
                print("Не удалось открыть видео!")

    def toggle_traffic_light_mode(self):
        if self.cap and self.cap.isOpened():
            self.traffic_light_mode = not self.traffic_light_mode
            if self.traffic_light_mode:
                self.btn_traffic_light.setText("Перейти в стандартный режим")
            else:
                self.btn_traffic_light.setText("Управление светофором")
            self.update_status()
        else:
            print("Сначала запустите видео или камеру!")

    def stop_video(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        self.image_label.clear()
        self.detected_labels.clear()
        self.spoken_labels.clear()
        self.positions.clear()
        self.speed_history.clear()
        self.violated_ids.clear()
        self.lane_violated_ids.clear()
        self.pedestrian_violated_ids.clear()
        self.red_light_violated_ids.clear()
        self.pedestrian_stop_counters.clear()
        self.previous_lanes.clear()
        self.lane_change_counters.clear()
        self.was_inside_crosswalk.clear()
        self.pedestrian_area = None
        self.traffic_light_mode = False
        self.btn_traffic_light.setText("Управление светофором")
        self.is_camera = False
        self.frame_id = 0
        self.update_status()

    def update_status(self):
        mode = "Светофор" if self.traffic_light_mode else "Стандартный"
        current_speed_limit = self.google_speed_limit if self.use_google_speed and self.google_speed_limit is not None else self.speed_limit
        self.status_label.setText(f"Устройство: {self.device} | FPS: {int(self.fps)} | Режим: {mode} | Ограничение скорости: {current_speed_limit} км/ч")

    def speak_label(self, label):
        self.speech_queue.put(label)

    def _speech_worker(self):
        while True:
            try:
                label = self.speech_queue.get()
                self.engine.say(label)
                self.engine.runAndWait()
                self.speech_queue.task_done()
            except Exception as e:
                print(f"Ошибка озвучивания: {e}")

    def init_db(self):
        self.conn = sqlite3.connect("violations.db")
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                timestamp TEXT,
                violation_type TEXT,
                plate_text TEXT,
                vehicle_path TEXT,
                plate_path TEXT
            )
        """)
        self.conn.commit()

    def log_violation(self, track_id, timestamp, violation_type, plate_text, vehicle_path, plate_path):
        self.cursor.execute("""
            INSERT INTO violations (track_id, timestamp, violation_type, plate_text, vehicle_path, plate_path)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (track_id, timestamp, violation_type, plate_text, vehicle_path, plate_path))
        self.conn.commit()

    def update_report(self):
        self.cursor.execute("SELECT violation_type, COUNT(*) as count FROM violations GROUP BY violation_type")
        results = self.cursor.fetchall()
        report_text = "Отчет о нарушениях:\n"
        for violation_type, count in results:
            report_text += f"{violation_type}: {count} раз\n"
        self.report_label.setText(report_text)

    def read_plate(self, plate_img_path):
        try:
            result = self.ocr_reader.readtext(plate_img_path)
            if result:
                plate_text = result[0][-2]
                return plate_text
            return None
        except Exception as e:
            print(f"Ошибка чтения номера: {e}")
            return None

    def detect_speed_violation(self, track_id, speed_kmh):
        current_speed_limit = self.google_speed_limit if self.use_google_speed and self.google_speed_limit is not None else self.speed_limit
        if speed_kmh > current_speed_limit:
            if track_id not in self.violated_ids:
                self.violated_ids.append(track_id)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"Нарушение скоростного режима: Автомобиль ID {track_id}, Скорость: {speed_kmh} км/ч, Ограничение: {current_speed_limit} км/ч")
                return True, timestamp
        return False, None

    def detect_red_light_violation(self, track_id, cx, cy, frame, x1, y1, x2, y2, detected_label):
        if detected_label != "RED":
            return False, None

        line_start, line_end = self.traffic_line
        x1_line, y1_line = line_start
        x2_line, y2_line = line_end

        if x2_line - x1_line != 0:
            m = (y2_line - y1_line) / (x2_line - x1_line)
            c = y1_line - m * x1_line
            line_y_at_cx = m * cx + c
            if cy > line_y_at_cx:
                if track_id not in self.red_light_violated_ids:
                    self.red_light_violated_ids.append(track_id)
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"Нарушение на красный свет: Автомобиль ID {track_id}")

                    vehicle_img = frame[y1:y2, x1:x2]
                    vehicle_filename = f"RedLight_ID_{track_id}_{timestamp.replace(' ', '_').replace(':', '-')}_vehicle.jpg"
                    vehicle_path = os.path.join(self.output_dir, vehicle_filename)
                    if vehicle_img.size > 0:
                        cv2.imwrite(vehicle_path, vehicle_img)

                    plate_y1 = y2 - int((y2 - y1) * 0.3)
                    plate_img = frame[plate_y1:y2, x1:x2]
                    plate_filename = f"RedLight_ID_{track_id}_{timestamp.replace(' ', '_').replace(':', '-')}_plate.jpg"
                    plate_path = os.path.join(self.output_dir, plate_filename)
                    if plate_img.size > 0:
                        cv2.imwrite(plate_path, plate_img)

                    self.add_violation_card(track_id, timestamp, "Нарушение на красный свет", vehicle_path, plate_path)
                    return True, timestamp
        return False, None

    def detect_lanes(self, frame):
        h, w = frame.shape[:2]
        roi = frame[h//4:h, 0:w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([255, 40, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        lower_yellow = np.array([10, 30, 140])
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.bitwise_and(edges, color_mask)
        edge_count = np.count_nonzero(edges)
        print(f"Количество пикселей после обнаружения краев: {edge_count}")
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=20)
        lane_lines = []
        if lines is not None:
            print(f"Обнаружены линии полос: найдено {len(lines)} линий")
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = abs((y2 - y1) / (x2 - x1 + 1e-5))
                if slope < 0.5:
                    continue
                y1_adjusted = y1 + h//4
                y2_adjusted = y2 + h//4
                lane_lines.append((x1, y1_adjusted, x2, y2_adjusted))
                cv2.line(frame, (x1, y1_adjusted), (x2, y2_adjusted), (255, 255, 0), 4)
        else:
            print("Линии полос не обнаружены!")
        return frame, lane_lines

    def detect_lane_violation(self, track_id, x1, x2, lane_lines):
        if not lane_lines:
            return False, None
        lane_x_positions = sorted([min(line[0], line[2]) for line in lane_lines])
        if len(lane_x_positions) < 2:
            return False, None
        vehicle_width = x2 - x1
        if vehicle_width <= 0:
            return False, None
        current_lane = None
        lane_portions = {}
        total_overlap = 0
        for i in range(len(lane_x_positions)):
            if i == 0:
                left_bound = 0
            else:
                left_bound = lane_x_positions[i - 1]
            if i == len(lane_x_positions) - 1:
                right_bound = 1920
            else:
                right_bound = lane_x_positions[i]
            left_bound_with_tolerance = left_bound + self.lane_tolerance
            right_bound_with_tolerance = right_bound - self.lane_tolerance
            overlap_start = max(x1, left_bound_with_tolerance)
            overlap_end = min(x2, right_bound_with_tolerance)
            overlap_width = max(0, overlap_end - overlap_start)
            portion = overlap_width / vehicle_width
            lane_portions[i] = portion
            total_overlap += portion
        if total_overlap > 0:
            dominant_lane = max(lane_portions, key=lane_portions.get)
            dominant_portion = lane_portions[dominant_lane]
            if dominant_portion >= self.lane_change_threshold:
                current_lane = dominant_lane
            else:
                current_lane = None
        if track_id in self.previous_lanes and current_lane is not None:
            prev_lane = self.previous_lanes[track_id]
            if current_lane != prev_lane:
                if track_id not in self.lane_change_counters:
                    self.lane_change_counters[track_id] = 0
                self.lane_change_counters[track_id] += 1
                if self.lane_change_counters[track_id] >= self.min_frames_for_lane_change:
                    if track_id not in self.lane_violated_ids:
                        self.lane_violated_ids.append(track_id)
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(f"Нарушение полосы: Автомобиль ID {track_id}, Полоса: {prev_lane} -> {current_lane}")
                        return True, timestamp
            else:
                self.lane_change_counters[track_id] = 0
        else:
            self.lane_change_counters[track_id] = 0
        if current_lane is not None:
            self.previous_lanes[track_id] = current_lane
        return False, None

    def detect_pedestrian_crosswalk(self, frame):
        h, w = frame.shape[:2]
        roi = frame[h//3:h, 0:w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([255, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        lower_yellow = np.array([15, 40, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)
        edges = cv2.bitwise_and(edges, color_mask)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        pedestrian_lines = []
        if lines is not None:
            print(f"Линии Hough для пешеходного перехода: найдено {len(lines)} линий")
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = abs((y2 - y1) / (x2 - x1 + 1e-5))
                if slope < 0.3:
                    y_avg = (y1 + y2) // 2
                    pedestrian_lines.append((x1, y_avg, x2, y_avg))
                    print(f"Горизонтальная линия: x1={x1}, y1={y1}, x2={x2}, y2={y2}, наклон={slope}")
        if len(pedestrian_lines) < 2:
            print("Линии пешеходного перехода не найдены: недостаточно линий обнаружено")
            return None
        pedestrian_lines = sorted(pedestrian_lines, key=lambda x: x[1])
        valid_lines = []
        for i in range(len(pedestrian_lines) - 1):
            line1 = pedestrian_lines[i]
            line2 = pedestrian_lines[i + 1]
            y_diff = abs(line2[1] - line1[1])
            if 10 < y_diff < 100:
                if not valid_lines or valid_lines[-1] != line1:
                    valid_lines.append(line1)
                valid_lines.append(line2)
        if len(valid_lines) < 2:
            print("Линии пешеходного перехода не найдены: нет линий с регулярным интервалом")
            return None
        min_y = min([line[1] for line in valid_lines]) + h//3
        max_y = max([line[1] for line in valid_lines]) + h//3
        min_x = min([min(line[0], line[2]) for line in valid_lines])
        max_x = max([max(line[0], line[2]) for line in valid_lines])
        pedestrian_area = [
            (min_x, min_y),
            (min_x, max_y),
            (max_x, max_y),
            (max_x, min_y)
        ]
        print(f"Создана зона пешеходного перехода: {pedestrian_area}")
        return pedestrian_area

    def detect_pedestrian_crosswalk_local(self, frame):
        results = self.model_crosswalk.predict(
            source=frame,
            conf=0.5,
            imgsz=640,
            show=False,
            verbose=False,
            device=self.device
        )
        pedestrian_area = None
        for r in results:
            for box in r.boxes:
                if r.names[int(box.cls)] == "crosswalk":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    pedestrian_area = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
                    print(f"Пешеходный переход обнаружен (локальная модель): {pedestrian_area}")
                    break
        return pedestrian_area

    def detect_pedestrians_in_area(self, frame, pedestrian_area, annotated_frame=None):
        if pedestrian_area is None:
            return False, []
        
        results = self.model_vehicles.predict(
            source=frame,
            conf=0.25,
            imgsz=640,
            show=False,
            verbose=False,
            device=self.device
        )
        has_pedestrian = False
        pedestrian_boxes = []
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    result = cv2.pointPolygonTest(np.array(pedestrian_area, np.int32), (cx, cy), False)
                    if result >= 0:
                        has_pedestrian = True
                        print(f"Обнаружен пешеход: cx={cx}, cy={cy}")
                        pedestrian_boxes.append((x1, y1, x2, y2))
                        if annotated_frame is not None:
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(annotated_frame, "Person", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return has_pedestrian, pedestrian_boxes

    def detect_pedestrian_violation(self, track_id, cx, cy, speed_kmh, pedestrian_area, has_pedestrian):
        if pedestrian_area is None or not has_pedestrian:
            return False, None
        result = cv2.pointPolygonTest(np.array(pedestrian_area, np.int32), (cx, cy), False)
        if result >= 0:
            if track_id not in self.pedestrian_stop_counters:
                self.pedestrian_stop_counters[track_id] = 0
            if speed_kmh <= 2:
                self.pedestrian_stop_counters[track_id] += 1
                if self.pedestrian_stop_counters[track_id] >= 10:
                    print(f"Автомобиль ID {track_id} остановился на пешеходном переходе, нарушения нет.")
                    return False, None
            else:
                if track_id not in self.pedestrian_violated_ids:
                    self.pedestrian_violated_ids.append(track_id)
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"Нарушение на пешеходном переходе (Скорость): Автомобиль ID {track_id}, Скорость: {speed_kmh} км/ч")
                    return True, timestamp
        else:
            self.pedestrian_stop_counters.pop(track_id, None)
        return False, None

    def detect_crosswalk_pass_violation(self, track_id, x1, y1, x2, y2, pedestrian_area, has_pedestrian, frame, vehicle_img):
        if pedestrian_area is None or not has_pedestrian:
            return False, None

        corners = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
        inside_count = sum(1 for corner in corners if cv2.pointPolygonTest(np.array(pedestrian_area, np.int32), corner, False) >= 0)

        is_inside = inside_count > 0

        was_inside = self.was_inside_crosswalk.get(track_id, False)

        self.was_inside_crosswalk[track_id] = is_inside

        if not is_inside and was_inside:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Нарушение на пешеходном переходе (Проезд): Автомобиль ID {track_id} полностью проехал пешеходный переход!")
            vehicle_filename = f"CrosswalkPass_ID_{track_id}_{timestamp.replace(' ', '_').replace(':', '-')}_vehicle.jpg"
            vehicle_path = os.path.join(self.output_dir, vehicle_filename)
            if vehicle_img.size > 0:
                cv2.imwrite(vehicle_path, vehicle_img)
            plate_y1 = y2 - int((y2 - y1) * 0.3)
            plate_img = frame[plate_y1:y2, x1:x2]
            plate_filename = f"CrosswalkPass_ID_{track_id}_{timestamp.replace(' ', '_').replace(':', '-')}_plate.jpg"
            plate_path = os.path.join(self.output_dir, plate_filename)
            if plate_img.size > 0:
                cv2.imwrite(plate_path, plate_img)
            self.add_violation_card(track_id, timestamp, "Нарушение на пешеходном переходе (Проезд)", vehicle_path, plate_path)
            return True, timestamp

        return False, None

    def add_violation_card(self, track_id, timestamp, violation_type, vehicle_path, plate_path):
        card = QWidget()
        card.setObjectName("violationCard")
        card_layout = QHBoxLayout()
        vehicle_label = QLabel()
        if os.path.exists(vehicle_path):
            pixmap = QPixmap(vehicle_path).scaled(100, 100, Qt.KeepAspectRatio)
            vehicle_label.setPixmap(pixmap)
        card_layout.addWidget(vehicle_label)
        plate_label = QLabel()
        if os.path.exists(plate_path):
            pixmap = QPixmap(plate_path).scaled(100, 100, Qt.KeepAspectRatio)
            plate_label.setPixmap(pixmap)
        card_layout.addWidget(plate_label)
        plate_text = self.read_plate(plate_path)
        info_widget = QWidget()
        info_layout = QVBoxLayout()
        info_label = QLabel(f"ID: {track_id}\nВремя: {timestamp}\nНарушение: {violation_type}\nНомер: {plate_text or 'неизвестно'}")
        info_label.setObjectName("violationLabel")
        info_layout.addWidget(info_label)
        info_widget.setLayout(info_layout)
        card_layout.addWidget(info_widget, stretch=1)
        card.setLayout(card_layout)
        card.setFixedHeight(120)
        self.violations_layout.addWidget(card)
        self.log_violation(track_id, timestamp, violation_type, plate_text, vehicle_path, plate_path)

    def clean_label(self, label):
        cleaned = label.replace("---", "--").split("--")
        cleaned = [part for part in cleaned if not part.startswith("g")]
        if len(cleaned) > 1:
            cleaned = cleaned[1:]
        cleaned_label = " ".join(cleaned).title()
        return cleaned_label

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.stop_video()
                return

            self.frame_id += 1
            frame = cv2.resize(frame, (1020, 600))
            original_h, original_w = frame.shape[:2]
            annotated_frame = frame.copy()

            if self.is_camera:
                self.image_label.setFixedSize(original_w, original_h)
                self.image_label.move(
                    (self.video_container.width() - original_w) // 2,
                    (self.video_container.height() - original_h) // 2
                )
            else:
                self.image_label.setFixedSize(self.video_container.size())
                self.image_label.setScaledContents(True)

            annotated_frame, lane_lines = self.detect_lanes(annotated_frame)

            if self.frame_id % self.pedestrian_check_interval == 0:
                self.pedestrian_area = self.detect_pedestrian_crosswalk_local(frame.copy())
                if self.pedestrian_area:
                    print("Пешеходный переход обнаружен (локальная модель)!")
                else:
                    print("Пешеходный переход не найден (локальная модель). Пробуем метод обработки изображения...")
                    self.pedestrian_area = self.detect_pedestrian_crosswalk(frame.copy())
                    if self.pedestrian_area:
                        print("Пешеходный переход обнаружен (обработка изображения)!")
                    else:
                        print("Пешеходный переход не найден (обработка изображения).")

            has_pedestrian, pedestrian_boxes = self.detect_pedestrians_in_area(frame.copy(), self.pedestrian_area, annotated_frame)

            if not self.traffic_light_mode:
                results_signs = self.model_signs.predict(
                    source=frame,
                    conf=0.5,
                    imgsz=640,
                    show=False,
                    verbose=False,
                    device=self.device
                )
                boxes_signs = results_signs[0].boxes
                labels_on_frame = set()

                for box in boxes_signs:
                    cls_id = int(box.cls[0])
                    label = self.model_signs.names[cls_id]
                    cleaned_label = self.clean_label(label)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = tuple(int(x) for x in np.random.default_rng(abs(hash(cleaned_label)) % (2**32)).integers(0, 255, size=3))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{cleaned_label}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(annotated_frame, text, (x1 + 2, y1 - 7),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    labels_on_frame.add(cleaned_label)
                    if cleaned_label not in self.spoken_labels:
                        self.spoken_labels.add(cleaned_label)
                        self.speak_label(cleaned_label)

                y_offset = 40
                for label in sorted(labels_on_frame):
                    color = tuple(int(x) for x in np.random.default_rng(abs(hash(label)) % (2**32)).integers(0, 255, size=3))
                    cv2.putText(annotated_frame, label,
                               (15, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                    y_offset += 40

                results_vehicles = self.model_vehicles.predict(
                    source=frame,
                    conf=0.25,
                    imgsz=640,
                    show=False,
                    verbose=False,
                    device=self.device
                )
                detections = []

                for r in results_vehicles:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        if cls in [2, 3, 5, 7]:
                            detections.append([x1, y1, x2, y2])

                detections = np.array(detections)
                if len(detections) > 0:
                    tracked_objects = self.tracker.update(detections)
                else:
                    tracked_objects = []

                for obj in tracked_objects:
                    x1, y1, x2, y2, track_id = obj
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    track_id = int(track_id)

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    lane_violation, lane_timestamp = self.detect_lane_violation(track_id, x1, x2, lane_lines)
                    if lane_violation:
                        vehicle_img = frame[y1:y2, x1:x2]
                        vehicle_filename = f"Lane_ID_{track_id}_{lane_timestamp.replace(' ', '_').replace(':', '-')}_vehicle.jpg"
                        vehicle_path = os.path.join(self.output_dir, vehicle_filename)
                        if vehicle_img.size > 0:
                            cv2.imwrite(vehicle_path, vehicle_img)
                        plate_y1 = y2 - int((y2 - y1) * 0.3)
                        plate_img = frame[plate_y1:y2, x1:x2]
                        plate_filename = f"Lane_ID_{track_id}_{lane_timestamp.replace(' ', '_').replace(':', '-')}_plate.jpg"
                        plate_path = os.path.join(self.output_dir, plate_filename)
                        if plate_img.size > 0:
                            cv2.imwrite(plate_path, plate_img)
                        self.add_violation_card(track_id, lane_timestamp, "Нарушение полосы", vehicle_path, plate_path)

                    vehicle_img = frame[y1:y2, x1:x2]
                    pass_violation, pass_timestamp = self.detect_crosswalk_pass_violation(
                        track_id, x1, y1, x2, y2, self.pedestrian_area, has_pedestrian, frame, vehicle_img
                    )

                    if pass_violation:
                        pass

                    if track_id not in self.trajectories:
                        self.trajectories[track_id] = deque(maxlen=30)
                    self.trajectories[track_id].append((cx, cy))

                    in_zone = (self.area[0][0] <= cx <= self.area[2][0] and self.area[0][1] <= cy <= self.area[2][1])

                    if in_zone and track_id in self.positions:
                        prev_cx, prev_cy, prev_frame = self.positions[track_id]
                        distance_pixels = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)

                        if distance_pixels > self.max_pixel_displacement:
                            distance_pixels = 0

                        distance_meters = distance_pixels * self.meters_per_pixel_base
                        time_seconds = (self.frame_id - prev_frame) / max(1, self.fps)

                        if time_seconds > 0:
                            speed_ms = distance_meters / time_seconds
                            speed_kmh = speed_ms * 3.6
                            if speed_kmh > self.max_speed_limit:
                                speed_kmh = self.max_speed_limit

                            if track_id not in self.speed_history:
                                self.speed_history[track_id] = deque(maxlen=self.max_speed_history)
                            self.speed_history[track_id].append(speed_kmh)

                            speed_violation, timestamp = self.detect_speed_violation(track_id, speed_kmh)
                            if speed_violation:
                                vehicle_img = frame[y1:y2, x1:x2]
                                vehicle_filename = f"Speed_ID_{track_id}_{timestamp.replace(' ', '_').replace(':', '-')}_vehicle.jpg"
                                vehicle_path = os.path.join(self.output_dir, vehicle_filename)
                                if vehicle_img.size > 0:
                                    cv2.imwrite(vehicle_path, vehicle_img)
                                plate_y1 = y2 - int((y2 - y1) * 0.3)
                                plate_img = frame[plate_y1:y2, x1:x2]
                                plate_filename = f"Speed_ID_{track_id}_{timestamp.replace(' ', '_').replace(':', '-')}_plate.jpg"
                                plate_path = os.path.join(self.output_dir, plate_filename)
                                if plate_img.size > 0:
                                    cv2.imwrite(plate_path, plate_img)
                                self.add_violation_card(track_id, timestamp, "Нарушение скоростного режима", vehicle_path, plate_path)

                            pedestrian_violation, ped_timestamp = self.detect_pedestrian_violation(
                                track_id, cx, cy, speed_kmh, self.pedestrian_area, has_pedestrian
                            )
                            if pedestrian_violation:
                                vehicle_img = frame[y1:y2, x1:x2]
                                vehicle_filename = f"Pedestrian_ID_{track_id}_{ped_timestamp.replace(' ', '_').replace(':', '-')}_vehicle.jpg"
                                vehicle_path = os.path.join(self.output_dir, vehicle_filename)
                                if vehicle_img.size > 0:
                                    cv2.imwrite(vehicle_path, vehicle_img)
                                plate_y1 = y2 - int((y2 - y1) * 0.3)
                                plate_img = frame[plate_y1:y2, x1:x2]
                                plate_filename = f"Pedestrian_ID_{track_id}_{ped_timestamp.replace(' ', '_').replace(':', '-')}_plate.jpg"
                                plate_path = os.path.join(self.output_dir, plate_filename)
                                if plate_img.size > 0:
                                    cv2.imwrite(plate_path, plate_img)
                                self.add_violation_card(track_id, ped_timestamp, "Нарушение на пешеходном переходе", vehicle_path, plate_path)

                    self.positions[track_id] = (cx, cy, self.frame_id)

                    hue = (track_id * 15) % 180
                    color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                    color = (int(color[0]), int(color[1]), int(color[2]))

                    for i in range(1, len(self.trajectories[track_id])):
                        cv2.line(annotated_frame, self.trajectories[track_id][i-1],
                                 self.trajectories[track_id][i], color, 2)

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    if in_zone and track_id in self.speed_history and len(self.speed_history[track_id]) > 0:
                        avg_speed = sum(self.speed_history[track_id]) / len(self.speed_history[track_id])
                        if avg_speed >= self.min_display_speed:
                            cv2.putText(annotated_frame, f"{int(avg_speed)} km/h", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            else:
                # Обработка светофора
                annotated_frame, detected_label = process_frame(frame.copy())
                self.traffic_light_indicator.setText(f"Светофор: {detected_label if detected_label else 'Неизвестно'}")

                if detected_label == "GREEN":
                    self.traffic_light_indicator.setStyleSheet("""
                        background-color: rgba(34, 197, 94, 0.9);
                        color: #E5E7EB;
                        font-size: 14px;
                        padding: 10px;
                        border-radius: 12px;
                        border: 1px solid #22C55E;
                    """)
                elif detected_label == "RED":
                    self.traffic_light_indicator.setStyleSheet("""
                        background-color: rgba(239, 68, 68, 0.9);
                        color: #E5E7EB;
                        font-size: 14px;
                        padding: 10px;
                        border-radius: 12px;
                        border: 1px solid #EF4444;
                    """)
                else:
                    self.traffic_light_indicator.setStyleSheet("""
                        background-color: rgba(63, 63, 90, 0.9);
                        color: #E5E7EB;
                        font-size: 14px;
                        padding: 10px;
                        border-radius: 12px;
                        border: 1px solid #6366F1;
                    """)

                # Линия для нарушения красного света
                cv2.line(annotated_frame, self.traffic_line[0], self.traffic_line[1], (0, 0, 255), 2)

                # Обнаружение транспортных средств
                results_vehicles = self.model_vehicles.predict(
                    source=frame,
                    conf=0.25,
                    imgsz=640,
                    show=False,
                    verbose=False,
                    device=self.device
                )
                detections = []

                for r in results_vehicles:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        if cls in [2, 3, 5, 7]:  # Транспортные средства
                            detections.append([x1, y1, x2, y2])

                detections = np.array(detections)
                if len(detections) > 0:
                    tracked_objects = self.tracker.update(detections)
                else:
                    tracked_objects = []

                for obj in tracked_objects:
                    x1, y1, x2, y2, track_id = obj
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    track_id = int(track_id)

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Проверка нарушения красного света
                    red_light_violation, red_light_timestamp = self.detect_red_light_violation(
                        track_id, cx, cy, frame, x1, y1, x2, y2, detected_label
                    )

                    # Отрисовка объекта
                    hue = (track_id * 15) % 180
                    color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                    color = (int(color[0]), int(color[1]), int(color[2]))

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Отрисовка зоны пешеходного перехода
            if self.pedestrian_area:
                cv2.polylines(annotated_frame, [np.array(self.pedestrian_area, np.int32)], True, (255, 0, 255), 3)
                cv2.putText(annotated_frame, "pedestrian crossing", (self.pedestrian_area[0][0], self.pedestrian_area[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Обновление отчета о нарушениях
            self.update_report()

            # Преобразование изображения для отображения
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap)

            # Обновление FPS
            current_time = datetime.now()
            elapsed = (current_time - self.last_fps_time).total_seconds()
            if elapsed > 1:
                self.fps = self.frame_id / elapsed if elapsed > 0 else 30
                self.frame_id = 0
                self.last_fps_time = current_time
                self.update_status()

    def closeEvent(self, event):
        self.stop_video()
        self.conn.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())