import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QLabel, QFileDialog, QHBoxLayout, QWidget)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO
import pyttsx3
import threading
import torch
from sort import Sort
import math
from collections import deque

class VideoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jol pril")
        
        # Начальный размер окна
        self.setMinimumSize(1366, 768)
        
        # Современный стиль
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E2E;
            }
            QPushButton {
                background-color: #6366F1;
                color: white;
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #818CF8;
            }
            QPushButton:pressed {
                background-color: #4F46E5;
            }
            QLabel#videoLabel {
                background-color: #2A2A3E;
                border: 3px solid #3F3F5A;
                border-radius: 10px;
            }
            QWidget#buttonContainer {
                background-color: #2A2A3E;
                border-top: 2px solid #3F3F5A;
            }
        """)

        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # YOLO modellerini GPU'da başlat
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Kullanılan cihaz: {self.device}")
        self.model_signs = YOLO("best.pt").to(self.device)  # Tabela modeli
        self.model_vehicles = YOLO("yolov8n.pt").to(self.device)  # Araç modeli

        # Ses motorunu başlat
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Konuşma hızı
        self.engine.setProperty('volume', 0.9)  # Ses seviyesi
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if "turkish" in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break

        # Okunan etiketleri takip et
        self.spoken_labels = set()

        # Araç takibi ve hız hesaplama için değişkenler
        self.tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.2)
        self.fps = 30  # Varsayılan FPS, kamera/video için güncellenecek
        self.meters_per_pixel_base = 0.07  # Kalibrasyon gerekli
        self.horizon_y = 50  # Kalibrasyon necessary
        self.perspective_factor = 1000
        self.positions = {}  # id -> (cx, cy, frame_id)
        self.speed_history = {}  # id -> deque
        self.max_speed_history = 10
        self.min_display_speed = 5
        self.max_speed_limit = 150
        self.max_pixel_displacement = 50

        # Интерфейс
        self.video_container = QWidget()
        self.video_container.setStyleSheet("background-color: #2A2A3E; border-radius: 10px;")

        # QLabel для видео
        self.image_label = QLabel(self.video_container)
        self.image_label.setObjectName("videoLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)

        # Кнопки
        self.btn_camera = QPushButton("Bilgisayar Kamerası")
        self.btn_phone_camera = QPushButton("Telefon Kamerası")
        self.btn_video = QPushButton("Видео")
        self.btn_stop = QPushButton("Стоп")
        self.btn_exit = QPushButton("Выйти")

        # Layout для кнопок
        h_layout = QHBoxLayout()
        for btn in [self.btn_camera, self.btn_phone_camera, self.btn_video, self.btn_stop, self.btn_exit]:
            h_layout.addWidget(btn)
            h_layout.setSpacing(15)

        # Контейнер для кнопок
        button_container = QWidget()
        button_container.setObjectName("buttonContainer")
        button_container.setLayout(h_layout)
        button_container.setFixedHeight(80)

        # Основной layout
        v_layout = QVBoxLayout()
        v_layout.addStretch(2)
        v_layout.addWidget(self.video_container, stretch=1)
        v_layout.addStretch(1)
        v_layout.addWidget(button_container)
        v_layout.setContentsMargins(20, 20, 20, 20)
        v_layout.setSpacing(0)
        
        central_widget.setLayout(v_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.cap = None
        self.detected_labels = set()
        self.is_camera = False
        self.frame_id = 0

        # Подключение сигналов
        self.btn_camera.clicked.connect(self.start_camera)
        self.btn_phone_camera.clicked.connect(self.start_phone_camera)
        self.btn_video.clicked.connect(self.open_video)
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_exit.clicked.connect(self.close)

    def resizeEvent(self, event):
        window_size = self.centralWidget().size()
        button_height = 80
        self.video_container.setFixedSize(
            window_size.width() - 40,
            window_size.height() - button_height - 40
        )
        super().resizeEvent(event)

    def start_camera(self):
        self.stop_video()
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.is_camera = True
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.timer.start(20)
        else:
            print("Bilgisayar kamerası açılamadı!")

    def start_phone_camera(self):
        self.stop_video()
        phone_camera_source = 1  # DroidCam cihaz indeksi
        # IP Webcam için: phone_camera_source = 'http://192.168.x.x:8080/video'
        self.cap = cv2.VideoCapture(phone_camera_source)
        if self.cap.isOpened():
            self.is_camera = True
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.timer.start(20)
        else:
            print("Telefon kamerası açılamadı! DroidCam/IP Webcam ayarlarını kontrol edin.")

    def open_video(self):
        self.stop_video()
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите видео")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            if self.cap.isOpened():
                self.is_camera = False
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
                self.timer.start(25)
            else:
                print("Video açılamadı!")

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
        self.is_camera = False
        self.frame_id = 0

    def speak_label(self, label):
        self.engine.say(label)
        self.engine.runAndWait()

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.stop_video()
                return

            self.frame_id += 1
            original_h, original_w = frame.shape[:2]

            if self.is_camera:
                self.image_label.setFixedSize(original_w, original_h)
                self.image_label.move(
                    (self.video_container.width() - original_w) // 2,
                    (self.video_container.height() - original_h) // 2
                )
            else:
                self.image_label.setFixedSize(self.video_container.size())
                self.image_label.setScaledContents(True)

            annotated_frame = frame.copy()

            # Tabela tespiti
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
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = tuple(int(x) for x in np.random.default_rng(abs(hash(label)) % (2**32)).integers(0, 255, size=3))

                # Рамка
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Подпись
                text = f"{label}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated_frame, text, (x1 + 2, y1 - 7),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                labels_on_frame.add(label)

                # Sesli okuma
                if label not in self.spoken_labels:
                    self.spoken_labels.add(label)
                    threading.Thread(target=self.speak_label, args=(label,), daemon=True).start()

            # Tabela etiketlerini ekranda göster
            y_offset = 40
            for label in sorted(labels_on_frame):
                color = tuple(int(x) for x in np.random.default_rng(abs(hash(label)) % (2**32)).integers(0, 255, size=3))
                cv2.putText(annotated_frame, label,
                           (15, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                y_offset += 40

            # Araç tespiti ve hız hesaplama
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
                    if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
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

                # Görselleştirme
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, f'ID:{track_id}', (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if track_id in self.speed_history and len(self.speed_history[track_id]) > 0:
                    avg_speed = int(sum(self.speed_history[track_id]) / len(self.speed_history[track_id]))
                    if avg_speed >= self.min_display_speed:
                        cv2.putText(annotated_frame, f"Speed: {avg_speed} km/h", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Конвертация для отображения
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoApp()
    player.show()
    sys.exit(app.exec_())