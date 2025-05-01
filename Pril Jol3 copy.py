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

        # Araç takibi ve hız hesaplama için değişkenler (mevcut değişkenler)
        self.tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.2)
        self.fps = 30  # Varsayılan FPS, kamera/video için güncellenecek
        self.meters_per_pixel_base = 0.07  # Kalibrasyon gerekli
        self.horizon_y = 50  # Kalibrasyon gerekli
        self.perspective_factor = 1000
        self.positions = {}  # id -> (cx, cy, frame_id)
        self.speed_history = {}  # id -> deque
        self.max_speed_history = 20  # detect_and_track.py'den artırılmış değer
        self.min_display_speed = 5
        self.max_speed_limit = 150
        self.max_pixel_displacement = 30  # detect_and_track.py'den

        # detect_and_track.py'den eklenen değişkenler
        self.real_distance_meters = 30  # Hız ölçüm bölgesi için gerçek mesafe (metre cinsinden)
        self.trajectories = {}  # id -> deque, araçların hareket yollarını tutmak için
        self.max_trajectory_length = 20  # Maksimum trajektuar uzunluğu

        # Hız ölçüm bölgesi (speed zone) tanımlama
        # Bu kısmı dinamik olarak ayarlayacağız, çünkü görüntü boyutu değişebilir
        self.speed_zone = None  # İlk başta None, update_frame'de dinamik olarak ayarlanacak

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

    # detect_and_track.py'den eklenen yardımcı fonksiyonlar
    def get_color(self, track_id):
        hue = (track_id * 15) % 180  # Her ID için farklı bir renk tonu
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        return (int(color[0]), int(color[1]), int(color[2]))

    def is_in_speed_zone(self, cx, cy, zone):
        return zone['x1'] <= cx <= zone['x2'] and zone['y1'] <= cy <= zone['y2']

    def put_text_with_background(self, frame, text, position, font_scale, color, thickness):
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        bg_x1, bg_y1 = position[0], position[1] - text_height - 5
        bg_x2, bg_y2 = position[0] + text_width + 5, position[1] + 5
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)  # Beyaz arka plan
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

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
        self.trajectories.clear()  # Yeni eklenen trajectories sıfırlanıyor
        self.speed_zone = None  # Hız ölçüm bölgesi sıfırlanıyor
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

            # Hız ölçüm bölgesini dinamik olarak tanımla
            if self.speed_zone is None:
                display_width, display_height = original_w, original_h
                self.speed_zone = {
                    'x1': int(display_width * 0.2),  # Sol sınır (20%)
                    'y1': int(display_height * 0.4),  # Üst sınır (40%)
                    'x2': int(display_width * 0.8),  # Sağ sınır (80%)
                    'y2': int(display_height * 0.7)  # Alt sınır (70%)
                }
                # Hız ölçüm bölgesinin piksel cinsinden uzunluğunu hesapla (dikey mesafe)
                zone_pixel_length = self.speed_zone['y2'] - self.speed_zone['y1']
                self.meters_per_pixel = self.real_distance_meters / zone_pixel_length  # Piksel başına metre

            # Hız ölçüm bölgesini çiz (yarı şeffaf dikdörtgen)
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (self.speed_zone['x1'], self.speed_zone['y1']),
                          (self.speed_zone['x2'], self.speed_zone['y2']),
                          (0, 255, 255), -1)  # Sarı dikdörtgen
            alpha = 0.1  # Şeffaflık
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

            # Bölge sınırlarını çiz (ince çizgiler)
            cv2.rectangle(annotated_frame, (self.speed_zone['x1'], self.speed_zone['y1']),
                          (self.speed_zone['x2'], self.speed_zone['y2']),
                          (0, 255, 255), 1)  # İnce sınır

            # Tabela tespiti (tamamen eski kod, hiçbir değişiklik yapılmadı)
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

            # Araç tespiti ve hız hesaplama (detect_and_track.py mantığı ile güncelleniyor)
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

                # Trajektuar oluşturma
                if track_id not in self.trajectories:
                    self.trajectories[track_id] = deque(maxlen=self.max_trajectory_length)
                self.trajectories[track_id].append((cx, cy))

                # Hız ölçüm bölgesinde mi kontrol et
                in_zone = self.is_in_speed_zone(cx, cy, self.speed_zone)
                if in_zone and track_id in self.positions:
                    prev_cx, prev_cy, prev_frame = self.positions[track_id]
                    # Sadece Y ekseninde mesafeyi hesapla (dikey hareket)
                    distance_pixels = abs(cy - prev_cy)
                    if distance_pixels < 5:  # Gürültü filtreleme
                        distance_pixels = 0
                    elif distance_pixels > self.max_pixel_displacement:  # Aşırı hareketleri filtrele
                        distance_pixels = 0

                    # Mesafeyi metreye çevir
                    distance_meters = distance_pixels * self.meters_per_pixel
                    time_seconds = (self.frame_id - prev_frame) / max(1, self.fps)
                    if time_seconds > 0:
                        speed_ms = distance_meters / time_seconds
                        speed_kmh = speed_ms * 3.6
                        if speed_kmh > self.max_speed_limit:
                            speed_kmh = self.max_speed_limit

                        if track_id not in self.speed_history:
                            self.speed_history[track_id] = deque(maxlen=self.max_speed_history)
                        if speed_kmh > 0:  # Sadece sıfır olmayan hızları ekle
                            self.speed_history[track_id].append(speed_kmh)

                self.positions[track_id] = (cx, cy, self.frame_id)

                # Görselleştirme
                color = self.get_color(track_id)
                # Trajektuar çizimi
                for i in range(1, len(self.trajectories[track_id])):
                    cv2.line(annotated_frame, self.trajectories[track_id][i-1],
                             self.trajectories[track_id][i], color, 2)

                # Araç etrafına ince çerçeve
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
                # ID'yi aracın altına yaz
                cv2.putText(annotated_frame, f'ID:{track_id}', (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Hız ölçüm bölgesindeyse hızı göster
                if in_zone and track_id in self.speed_history and len(self.speed_history[track_id]) > 0:
                    avg_speed = int(sum(self.speed_history[track_id]) / len(self.speed_history[track_id]))
                    if avg_speed >= self.min_display_speed:
                        self.put_text_with_background(annotated_frame, f"{avg_speed}km/h",
                                                     (x1, y1 - 10), 0.7, color, 2)

            # Görüntüyü PyQt için dönüştür
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