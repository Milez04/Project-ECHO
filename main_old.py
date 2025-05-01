import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QLabel, QFileDialog, QWidget, QTabWidget, QScrollArea, QGridLayout)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QFontDatabase
from PyQt5.QtCore import QTimer, Qt, QSize, QPropertyAnimation, QEasingCurve
from ultralytics import YOLO
import pyttsx3
import threading
import torch
from sort import Sort
import math
from collections import deque
import os
from datetime import datetime

# process_frame fonksiyonu
def process_frame(frame):
    lower_range = np.array([58, 97, 222])  # Green color range
    upper_range = np.array([179, 255, 255])
    lower_range1 = np.array([0, 43, 184])  # Red color range
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
        self.setWindowTitle("Жол қосымшасы")
        self.setMinimumSize(1366, 768)

        # Font yükleme
        QFontDatabase().addApplicationFont("fonts/Inter-Regular.ttf")
        QFontDatabase().addApplicationFont("fonts/Inter-Bold.ttf")

        # Stil sayfası
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
                transform: scale(1.05);
            }
            QPushButton:pressed {
                background: #4F46E5;
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
        print(f"Қолданылған құрылғы: {self.device}")
        self.model_signs = YOLO("best.pt").to(self.device)
        self.model_vehicles = YOLO("yolov8n.pt").to(self.device)

        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if "turkish" in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break

        self.spoken_labels = set()

        self.tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.2)
        self.fps = 30
        self.meters_per_pixel_base = 0.07
        self.horizon_y = 50
        self.perspective_factor = 1000
        self.positions = {}
        self.speed_history = {}
        self.max_speed_history = 10
        self.min_display_speed = 5
        self.max_speed_limit = 150
        self.max_pixel_displacement = 50

        self.area = [(324, 313), (283, 374), (854, 392), (864, 322)]
        self.violated_ids = []
        self.today_date = datetime.now().strftime('%Y-%m-%d')
        self.output_dir = os.path.join('saved_images', self.today_date)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.traffic_light_mode = False

        # Ana düzen
        main_layout = QHBoxLayout()

        # Sol sidebar
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(280)  # Sidebar genişliğini artırdık
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setAlignment(Qt.AlignTop)
        sidebar_layout.setSpacing(10)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)

        # Logo ve slogan
        logo_label = QLabel("Жол қосымшасы")
        logo_label.setStyleSheet("color: #E5E7EB; font-size: 20px; font-weight: bold;")
        slogan_label = QLabel("Қауіпсіз жолдар үшін")
        slogan_label.setStyleSheet("color: #A5B4FC; font-size: 12px;")
        sidebar_layout.addWidget(logo_label)
        sidebar_layout.addWidget(slogan_label)
        sidebar_layout.addSpacing(20)

        # Butonlar
        self.btn_camera = QPushButton("Компьютер камерасы")
        self.btn_camera.setIcon(QIcon("icons/camera.svg"))
        self.btn_phone_camera = QPushButton("Телефон камерасы")
        self.btn_phone_camera.setIcon(QIcon("icons/phone.svg"))
        self.btn_video = QPushButton("Бейне")
        self.btn_video.setIcon(QIcon("icons/video.svg"))
        self.btn_traffic_light = QPushButton("Бағдаршамды басқару")
        self.btn_traffic_light.setIcon(QIcon("icons/traffic-light.svg"))
        self.btn_stop = QPushButton("Тоқтату")
        self.btn_stop.setIcon(QIcon("icons/stop.svg"))
        self.btn_exit = QPushButton("Шығу")
        self.btn_exit.setIcon(QIcon("icons/exit.svg"))

        for btn in [self.btn_camera, self.btn_phone_camera, self.btn_video, self.btn_traffic_light, self.btn_stop, self.btn_exit]:
            btn.setFixedHeight(50)
            # setWordWrap kaldırıldı, stil ile metin sığdırılacak
            sidebar_layout.addWidget(btn)

        sidebar.setLayout(sidebar_layout)
        main_layout.addWidget(sidebar)

        # Sağ içerik alanı
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(20, 20, 20, 20)

        # Durum çubuğu
        self.status_label = QLabel(f"Құрылғы: {self.device} | FPS: 0 | Режим: Стандартты")
        self.status_label.setObjectName("statusLabel")
        content_layout.addWidget(self.status_label)

        # Sekmeler
        self.tab_widget = QTabWidget()
        self.video_tab = QWidget()
        self.violations_tab = QWidget()

        self.tab_widget.addTab(self.video_tab, "Бейне ағыны")
        self.tab_widget.addTab(self.violations_tab, "Қызыл шам бұзушылықтары")

        # Video sekmesi
        video_layout = QVBoxLayout()
        self.video_container = QWidget()
        self.video_container.setStyleSheet("background-color: #2A2A3E; border-radius: 12px; position: relative;")
        self.image_label = QLabel()
        self.image_label.setObjectName("videoLabel")
        self.image_label.setAlignment(Qt.AlignCenter)

        # Trafik ışığı göstergesi (video çerçevesinin üzerine)
        self.traffic_light_indicator = QLabel("Бағдаршам: Белгісіз")
        self.traffic_light_indicator.setObjectName("trafficLightIndicator")
        self.traffic_light_indicator.setFixedSize(200, 40)
        self.traffic_light_indicator.setParent(self.video_container)
        self.traffic_light_indicator.move(10, 10)

        video_container_layout = QVBoxLayout()
        video_container_layout.addWidget(self.image_label)
        self.video_container.setLayout(video_container_layout)
        video_layout.addWidget(self.video_container, stretch=1)

        self.video_tab.setLayout(video_layout)

        # İhlal sekmesi
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

        content_layout.addWidget(self.tab_widget)
        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget, stretch=1)

        central_widget.setLayout(main_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.cap = None
        self.detected_labels = set()
        self.is_camera = False
        self.frame_id = 0
        self.last_fps_time = datetime.now()

        self.btn_camera.clicked.connect(self.start_camera)
        self.btn_phone_camera.clicked.connect(self.start_phone_camera)
        self.btn_video.clicked.connect(self.open_video)
        self.btn_traffic_light.clicked.connect(self.toggle_traffic_light_mode)
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_exit.clicked.connect(self.close)

    def resizeEvent(self, event):
        window_size = self.centralWidget().size()
        self.video_container.setFixedSize(int(window_size.width() * 0.85), window_size.height() - 100)
        indicator_width = self.traffic_light_indicator.width()
        self.traffic_light_indicator.move(self.video_container.width() - indicator_width - 10, 10)
        super().resizeEvent(event)

    def start_camera(self):
        self.traffic_light_mode = False
        self.btn_traffic_light.setText("Бағдаршамды басқару")
        self.stop_video()
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.is_camera = True
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.timer.start(20)
        else:
            print("Компьютер камерасы ашылмады!")

    def start_phone_camera(self):
        self.traffic_light_mode = False
        self.btn_traffic_light.setText("Бағдаршамды басқару")
        self.stop_video()
        phone_camera_source = 1
        self.cap = cv2.VideoCapture(phone_camera_source)
        if self.cap.isOpened():
            self.is_camera = True
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.timer.start(20)
        else:
            print("Телефон камерасы ашылмады! DroidCam/IP Webcam параметрлерін тексеріңіз.")

    def open_video(self):
        self.traffic_light_mode = False
        self.btn_traffic_light.setText("Бағдаршамды басқару")
        self.stop_video()
        file_path, _ = QFileDialog.getOpenFileName(self, "Бейне таңдаңыз")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            if self.cap.isOpened():
                self.is_camera = False
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
                self.timer.start(25)
            else:
                print("Бейне ашылмады!")

    def toggle_traffic_light_mode(self):
        if self.cap and self.cap.isOpened():
            self.traffic_light_mode = not self.traffic_light_mode
            if self.traffic_light_mode:
                self.btn_traffic_light.setText("Стандартты режимге өту")
            else:
                self.btn_traffic_light.setText("Бағдаршамды басқару")
                self.positions.clear()
                self.speed_history.clear()
            self.update_status()
        else:
            print("Алдымен бейне немесе камераны іске қосыңыз!")

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
        self.traffic_light_mode = False
        self.btn_traffic_light.setText("Бағдаршамды басқару")
        self.is_camera = False
        self.frame_id = 0
        self.update_status()

    def update_status(self):
        mode = "Бағдаршам" if self.traffic_light_mode else "Стандартты"
        self.status_label.setText(f"Құрылғы: {self.device} | FPS: {int(self.fps)} | Режим: {mode}")

    def speak_label(self, label):
        self.engine.say(label)
        self.engine.runAndWait()

    def add_violation_card(self, track_id, timestamp, vehicle_path, plate_path):
        card = QWidget()
        card.setObjectName("violationCard")
        card_layout = QHBoxLayout()

        # Araç fotoğrafı
        vehicle_label = QLabel()
        if os.path.exists(vehicle_path):
            pixmap = QPixmap(vehicle_path).scaled(100, 100, Qt.KeepAspectRatio)
            vehicle_label.setPixmap(pixmap)
        card_layout.addWidget(vehicle_label)

        # Plaka fotoğrafı
        plate_label = QLabel()
        if os.path.exists(plate_path):
            pixmap = QPixmap(plate_path).scaled(100, 100, Qt.KeepAspectRatio)
            plate_label.setPixmap(pixmap)
        card_layout.addWidget(plate_label)

        # Bilgi
        info_widget = QWidget()
        info_layout = QVBoxLayout()
        info_label = QLabel(f"ID: {track_id}\nУақыт: {timestamp}\nҚызыл шамда өтті")
        info_label.setObjectName("violationLabel")
        info_layout.addWidget(info_label)
        info_widget.setLayout(info_layout)
        card_layout.addWidget(info_widget, stretch=1)

        card.setLayout(card_layout)
        card.setFixedHeight(120)
        self.violations_layout.addWidget(card)

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
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    color = tuple(int(x) for x in np.random.default_rng(abs(hash(label)) % (2**32)).integers(0, 255, size=3))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{label}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(annotated_frame, text, (x1 + 2, y1 - 7),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    labels_on_frame.add(label)

                    if label not in self.spoken_labels:
                        self.spoken_labels.add(label)
                        threading.Thread(target=self.speak_label, args=(label,), daemon=True).start()

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

                    if track_id in self.positions:
                        prev_cx, prev_cy, prev_frame = self.positions[track_id]
                        distance_pixels = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                        if distance_pixels < 5:
                            distance_pixels = 0
                        elif distance_pixels > self.max_pixel_displacement:
                            distance_pixels = 0

                        if cy > self.horizon_y:
                            perspective_scale = self.perspective_factor / (self.perspective_factor + (cy - self.horizon_y))
                            adjusted_meters_per_pixel = self.meters_per_pixel_base / perspective_scale
                        else:
                            adjusted_meters_per_pixel = self.meters_per_pixel_base * 10

                        distance_meters = distance_pixels * adjusted_meters_per_pixel
                        time_seconds = (self.frame_id - prev_frame) / self.fps
                        if time_seconds > 0:
                            speed_ms = distance_meters / time_seconds
                            speed_kmh = speed_ms * 3.6
                            if speed_kmh > self.max_speed_limit:
                                speed_kmh = self.max_speed_limit

                            if track_id not in self.speed_history:
                                self.speed_history[track_id] = deque(maxlen=self.max_speed_history)
                            if speed_kmh > 0:
                                self.speed_history[track_id].append(speed_kmh)

                            if len(self.speed_history[track_id]) > 0:
                                avg_speed = sum(self.speed_history[track_id]) / len(self.speed_history[track_id])
                                avg_speed = max(0, min(self.max_speed_limit, avg_speed))

                    self.positions[track_id] = (cx, cy, self.frame_id)

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f'ID:{track_id}', (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    if track_id in self.speed_history and len(self.speed_history[track_id]) > 0:
                        avg_speed = int(sum(self.speed_history[track_id]) / len(self.speed_history[track_id]))
                        if avg_speed >= self.min_display_speed:
                            cv2.putText(annotated_frame, f"Speed: {avg_speed} km/h", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                processed_frame, detected_label = process_frame(annotated_frame.copy())
                if detected_label:
                    print(f"Бағдаршам: {detected_label}")
                    self.traffic_light_indicator.setText(f"Бағдаршам: {detected_label}")
                    self.traffic_light_indicator.setStyleSheet(f"""
                        background-color: rgba({'255, 85, 85, 0.9' if detected_label == 'RED' else '85, 255, 85, 0.9'});
                        color: #FFFFFF;
                        font-size: 14px;
                        padding: 10px;
                        border-radius: 12px;
                        border: 1px solid #6366F1;
                    """)
                else:
                    self.traffic_light_indicator.setText("Бағдаршам: Белгісіз")
                    self.traffic_light_indicator.setStyleSheet("""
                        background-color: rgba(63, 63, 90, 0.9);
                        color: #E5E7EB;
                        font-size: 14px;
                        padding: 10px;
                        border-radius: 12px;
                        border: 1px solid #6366F1;
                    """)
                annotated_frame = processed_frame

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
                        if cls == 2:
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

                    cv2.circle(annotated_frame, (cx, cy), 4, (255, 0, 0), -1)
                    result = cv2.pointPolygonTest(np.array(self.area, np.int32), ((cx, cy)), False)

                    if result >= 0 and detected_label == "RED":
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated_frame, f'ID:{track_id}', (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        if track_id not in self.violated_ids:
                            self.violated_ids.append(track_id)
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            timestamp_clean = timestamp.replace(' ', '_').replace(':', '-')

                            vehicle_img = frame[y1:y2, x1:x2]
                            vehicle_filename = f"ID_{track_id}_{timestamp_clean}_vehicle.jpg"
                            vehicle_path = os.path.join(self.output_dir, vehicle_filename)
                            if vehicle_img.size > 0:
                                cv2.imwrite(vehicle_path, vehicle_img)

                            plate_y1 = y2 - int((y2 - y1) * 0.3)
                            plate_img = frame[plate_y1:y2, x1:x2]
                            plate_filename = f"ID_{track_id}_{timestamp_clean}_plate.jpg"
                            plate_path = os.path.join(self.output_dir, plate_filename)
                            if plate_img.size > 0:
                                cv2.imwrite(plate_path, plate_img)

                            self.add_violation_card(track_id, timestamp, vehicle_path, plate_path)

                            threading.Thread(target=self.speak_label, args=("Қызыл шам бұзушылығы!",), daemon=True).start()
                    else:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f'ID:{track_id}', (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.polylines(annotated_frame, [np.array(self.area, np.int32)], True, (0, 255, 0), 2)

            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap)

            # FPS güncelleme
            current_time = datetime.now()
            elapsed = (current_time - self.last_fps_time).total_seconds()
            if elapsed > 1:
                self.fps = self.frame_id / elapsed
                self.frame_id = 0
                self.last_fps_time = current_time
                self.update_status()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont("Inter", 10))
    player = VideoApp()
    player.show()
    sys.exit(app.exec_())