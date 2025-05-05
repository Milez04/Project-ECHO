import cv2
import numpy as np
import pandas as pd
import datetime
import time
from ultralytics import YOLO
import face_recognition
import easyocr
import sqlite3
from collections import defaultdict
import pywhatkit
import os
import sys
import warnings

# PyQt5 kütüphaneleri
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QIcon, QLinearGradient
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, 
                            QPushButton, QScrollArea, QFrame, QSplitter, QTabWidget, QTableWidget, 
                            QTableWidgetItem, QHeaderView, QStatusBar, QMessageBox, QSizePolicy, QFileDialog)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QSize

# SORT tracker
from sort import Sort

# Uyarılardan kaçın
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Qt ölçeklendirme uyarısını çöz
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

# Stil için renkler
COLORS = {
    'primary': '#4a90e2',
    'secondary': '#50e3c2',
    'accent': '#f5a623',
    'warning': '#ff2d55',
    'danger': '#d81b60',
    'success': '#2ecc71',
    'dark': '#1a252f',
    'light': '#ffffff',
    'white': '#ffffff',
    'black': '#000000',
    'gray': '#bdc3c7'
}

class ProcessingThread(QThread):
    image_update = pyqtSignal(QImage)
    status_update = pyqtSignal(str)
    violation_detected = pyqtSignal(dict)
    db_query = pyqtSignal(str, tuple, str)
    save_violation_image = pyqtSignal(np.ndarray, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.cap = None
        self.device = 'cpu'
        
        try:
            self.status_update.emit("Загружается модель YOLO...")
            self.plate_model = YOLO("yolov8s.pt").to(self.device)
            self.status_update.emit("Модель YOLO загружена")
        except Exception as e:
            self.status_update.emit(f"Ошибка загрузки YOLO: {e}")
            print(f"Ошибка загрузки YOLO: {e}")
        
        try:
            self.status_update.emit("Загружается модель OCR...")
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            self.status_update.emit("Модель OCR загружена")
        except Exception as e:
            self.status_update.emit(f"Ошибка загрузки OCR: {e}")
            print(f"Ошибка загрузки OCR: {e}")
        
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.2)
        
        self.processed_plates = defaultdict(lambda: {'last_processed': None, 'count': 0, 'status': None, 'owner': ''})
        
        self.face_frame_counter = 0
        self.face_process_interval = 5
        
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
        self.tracked_objects = {}
        self.last_detected_person = None  # Последний обнаруженный человек
        
    def load_known_faces(self):
        try:
            if not os.path.exists("faces"):
                os.makedirs("faces")
                self.status_update.emit("Папка 'faces' создана")
                print("Папка 'faces' создана")
                
            face_files = os.listdir("faces")
            print(f"Файлы в папке faces: {face_files}")
            
            for face_file in face_files:
                if face_file.endswith((".jpg", ".png")):
                    try:
                        known_image = face_recognition.load_image_file(os.path.join("faces", face_file))
                        if known_image.size > 0:
                            face_encodings = face_recognition.face_encodings(known_image)
                            if face_encodings:
                                self.known_face_encodings.append(face_encodings[0])
                                self.known_face_names.append(face_file.split('.')[0])
                                print(f"Лицо загружено: {face_file}")
                            else:
                                print(f"Кодирование лица не найдено: {face_file}")
                        else:
                            print(f"Недопустимое изображение: {face_file}")
                    except Exception as e:
                        print(f"Ошибка загрузки лица ({face_file}): {e}")
            
            self.status_update.emit(f"Загружено лиц: {len(self.known_face_names)}")
            print(f"Загруженные лица: {self.known_face_names}")
        except Exception as e:
            self.status_update.emit(f"Ошибка загрузки лиц: {e}")
            print(f"Ошибка загрузки лиц: {e}")
            
    def set_camera(self, source):
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(source)
            print(f"VideoCapture открыт? {self.cap.isOpened()} (Источник: {source})")
            if not self.cap.isOpened():
                self.status_update.emit(f"Источник не открыт: {source}")
                print(f"Источник не открыт: {source}")
                return False
            return True
        except Exception as e:
            self.status_update.emit(f"Ошибка открытия источника: {e}")
            print(f"Ошибка открытия источника: {e}")
            return False
    
    def run(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None or frame.size == 0:
                    self.status_update.emit("Кадр не получен! Видео может быть завершено.")
                    print("Ошибка: Кадр не получен! Видео может быть завершено.")
                    break
                
                print("Кадр получен, обрабатывается...")
                
                results = self.plate_model(frame)
                detections = []
                for r in results:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy()
                    for box, conf, cls in zip(boxes, confs, classes):
                        if conf > 0.5 and cls == 2:  # Class 2: Car
                            detections.append([box[0], box[1], box[2], box[3], conf])
                
                            print("Обнаружение номера YOLO завершено.")
                
                if len(detections) > 0:
                    tracks = self.tracker.update(np.array(detections))
                else:
                    tracks = self.tracker.update(np.empty((0, 5)))
                
                print("Отслеживание SORT завершено.")
                
                current_time = time.time()
                expired_ids = [tid for tid, data in self.tracked_objects.items() 
                              if current_time - data.get('timestamp', 0) > 5]
                for tid in expired_ids:
                    del self.tracked_objects[tid]
                
                for track in tracks:
                    x1, y1, x2, y2, track_id = track[:5]
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    plate_region = frame[y1:y2, x1:x2]
                    if plate_region.size == 0:
                        continue
                    
                    print(f"Track ID {track_id}: Область номера получена.")
                    
                    ocr_results = self.ocr_reader.readtext(plate_region)
                    plate_text = ""
                    for (bbox, text, prob) in ocr_results:
                        if prob > 0.5:
                            plate_text += text + " "
                    plate_text = plate_text.strip().replace(" ", "")
                    
                    print(f"Track ID {track_id}: Номер прочитан: {plate_text}")
                    
                    if plate_text:
                        self.tracked_objects[track_id] = self.tracked_objects.get(track_id, {})
                        self.tracked_objects[track_id].update({
                            'plate': {'coords': (x1, y1, x2, y2), 'text': plate_text, 'timestamp': current_time}
                        })
                        self.processed_plates[plate_text].update({
                            'last_processed': datetime.datetime.now(),
                            'count': self.processed_plates[plate_text]['count'] + 1
                        })
                    
                    if plate_text:
                        violation_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        violation_image_path = f"violations/violation_{track_id}_{violation_time.replace(':', '-')}.jpg"
                        self.db_query.emit(
                            "SELECT * FROM prapiska WHERE Plaka = ?",
                            (plate_text,),
                            "select"
                        )
                        self.save_violation_image.emit(frame, violation_image_path)
                
                face_locations = []
                face_names = []
                if self.face_frame_counter % self.face_process_interval == 0:
                    print("Начинается распознавание лиц...")
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    face_names = []
                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        name = "Darhan"
                        if True in matches:
                            first_match_index = matches.index(True)
                            name = self.known_face_names[first_match_index]
                        face_names.append(name)
                        print(f"Обнаруженное лицо: {name}")
                    
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        face_id = f"face_{hash(str((top, right, bottom, left)))}"
                        self.tracked_objects[face_id] = {
                            'face': {'coords': (left, top, right, bottom), 'name': name, 'timestamp': current_time}
                        }
                        self.last_detected_person = name  # Обновляем последнего обнаруженного человека
                        print(f"Последний обнаруженный человек обновлен: {self.last_detected_person}")
                
                self.face_frame_counter += 1
                
                for track_id, data in self.tracked_objects.items():
                    if 'plate' in data:
                        x1, y1, x2, y2 = data['plate']['coords']
                        plate_text = data['plate']['text']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, f"Plate: {plate_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
                    if 'face' in data:
                        left, top, right, bottom = data['face']['coords']
                        name = data['face']['name']
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                self.image_update.emit(q_img)
                
                print("Изображение отправлено.")
            
            except Exception as e:
                self.status_update.emit(f"Ошибка обработки: {e}")
                print(f"Ошибка обработки: {e}")
                break
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.wait()


class VideoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Система распознавания номеров и контроля прописки")
        self.setGeometry(100, 50, 1400, 800)
        self.setMinimumSize(1200, 700)
        
        self.setup_ui()
        self.setup_database()
        
        self.processing_thread = ProcessingThread()
        self.processing_thread.image_update.connect(self.update_frame)
        self.processing_thread.status_update.connect(self.update_status)
        self.processing_thread.violation_detected.connect(self.add_violation)
        self.processing_thread.db_query.connect(self.handle_db_query)
        self.processing_thread.save_violation_image.connect(self.save_violation_image)
        
        self.output_dir = "violations"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.start_camera()
    
    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Градиентный фон
        gradient = QLinearGradient(0, 0, 0, 400)
        gradient.setColorAt(0, QColor(COLORS['dark']))
        gradient.setColorAt(1, QColor(COLORS['primary']))
        palette = self.palette()
        palette.setBrush(QPalette.Window, gradient)
        self.setPalette(palette)
        
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        header_left = QWidget()
        header_left_layout = QHBoxLayout(header_left)
        header_left_layout.setContentsMargins(10, 10, 10, 10)
        
        logo_label = QLabel()
        logo_pixmap = QPixmap("logo.png") if os.path.exists("logo.png") else QPixmap(32, 32)
        if logo_pixmap.isNull():
            logo_pixmap.fill(QColor(COLORS['accent']))
        logo_label.setPixmap(logo_pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        header_left_layout.addWidget(logo_label)
        
        title_label = QLabel("Продвинутая система распознавания номеров")
        title_font = QFont("Arial", 20, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"color: {COLORS['white']};")
        header_left_layout.addWidget(title_label)
        header_left_layout.addStretch()
        
        header_layout.addWidget(header_left, 7)
        
        header_right = QWidget()
        header_right_layout = QHBoxLayout(header_right)
        header_right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.date_time_label = QLabel(datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
        self.date_time_label.setStyleSheet(f"color: {COLORS['white']}; font-size: 14px; font-weight: bold;")
        header_right_layout.addWidget(self.date_time_label)
        
        self.datetime_timer = QTimer(self)
        self.datetime_timer.timeout.connect(self.update_datetime)
        self.datetime_timer.start(1000)
        
        header_layout.addWidget(header_right, 3)
        
        main_layout.addLayout(header_layout)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {COLORS['gray']}; width: 5px; }}")
        splitter.setHandleWidth(5)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        self.camera_frame = QLabel()
        self.camera_frame.setAlignment(Qt.AlignCenter)
        self.camera_frame.setMinimumSize(800, 600)
        self.camera_frame.setMaximumSize(800, 600)
        self.camera_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.camera_frame.setStyleSheet(f"""
            background-color: {COLORS['light']}; 
            border-radius: 15px;
            border: 3px solid {COLORS['accent']};
        """)
        left_layout.addWidget(self.camera_frame)
        
        camera_controls = QHBoxLayout()
        camera_controls.setSpacing(10)
        
        self.start_button = QPushButton("Запустить")
        self.start_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.start_button.clicked.connect(self.start_camera)
        self.start_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: {COLORS['white']};
                border: none;
                border-radius: 10px;
                padding: 12px 25px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #27ae60;
                transform: scale(1.05);
                transition: all 0.3s ease;
            }}
        """)
        
        self.stop_button = QPushButton("Остановить")
        self.stop_button.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['danger']};
                color: {COLORS['white']};
                border: none;
                border-radius: 10px;
                padding: 12px 25px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #c0392b;
                transform: scale(1.05);
                transition: all 0.3s ease;
            }}
        """)
        
        self.select_video_button = QPushButton("Выбрать видео")
        self.select_video_button.setIcon(QIcon.fromTheme("folder-videos"))
        self.select_video_button.clicked.connect(self.select_video)
        self.select_video_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: {COLORS['white']};
                border: none;
                border-radius: 10px;
                padding: 12px 25px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #3b79c3;
                transform: scale(1.05);
                transition: all 0.3s ease;
            }}
        """)
        
        camera_controls.addWidget(self.start_button)
        camera_controls.addWidget(self.stop_button)
        camera_controls.addWidget(self.select_video_button)
        camera_controls.addStretch()
        
        left_layout.addLayout(camera_controls)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{ 
                border: none;
                background-color: {COLORS['light']};
                border-radius: 15px;
            }}
            QTabBar::tab {{ 
                background-color: {COLORS['dark']}; 
                color: {COLORS['white']}; 
                padding: 12px 25px;
                margin-right: 5px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
            }}
            QTabBar::tab:selected {{ 
                background-color: {COLORS['primary']}; 
                color: {COLORS['white']};
            }}
        """)
        
        violations_tab = QWidget()
        violations_layout = QVBoxLayout(violations_tab)
        violations_layout.setContentsMargins(15, 15, 15, 15)
        
        violations_header = QHBoxLayout()
        violations_title = QLabel("Отчеты о нарушениях")
        violations_title.setFont(QFont("Arial", 18, QFont.Bold))
        violations_title.setStyleSheet(f"color: {COLORS['danger']};")
        violations_header.addWidget(violations_title)
        
        self.violation_counter = QLabel("Всего: 0")
        self.violation_counter.setStyleSheet(f"color: {COLORS['warning']}; font-size: 16px; font-weight: bold;")
        violations_header.addWidget(self.violation_counter, alignment=Qt.AlignRight)
        
        violations_layout.addLayout(violations_header)
        
        self.violations_scroll = QScrollArea()
        self.violations_scroll.setWidgetResizable(True)
        self.violations_widget = QWidget()
        self.violations_layout = QVBoxLayout(self.violations_widget)
        self.violations_layout.setAlignment(Qt.AlignTop)
        self.violations_layout.setSpacing(10)
        self.violations_scroll.setStyleSheet(f"""
            QScrollArea {{ 
                background-color: {COLORS['light']}; 
                border: none;
                border-radius: 15px;
            }}
        """)
        self.violations_scroll.setWidget(self.violations_widget)
        violations_layout.addWidget(self.violations_scroll)
        
        self.clear_violations_btn = QPushButton("Очистить нарушения")
        self.clear_violations_btn.setIcon(QIcon.fromTheme("edit-clear"))
        self.clear_violations_btn.clicked.connect(self.clear_violations)
        self.clear_violations_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['warning']};
                color: {COLORS['white']};
                border: none;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #e67e22;
                transform: scale(1.05);
                transition: all 0.3s ease;
            }}
        """)
        violations_layout.addWidget(self.clear_violations_btn)
        
        vehicles_tab = QWidget()
        vehicles_layout = QVBoxLayout(vehicles_tab)
        vehicles_layout.setContentsMargins(15, 15, 15, 15)
        
        self.vehicles_table = QTableWidget(0, 4)
        self.vehicles_table.setHorizontalHeaderLabels(["Номер", "Статус", "Владелец", "Последнее обнаружение"])
        self.vehicles_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.vehicles_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.vehicles_table.setAlternatingRowColors(True)
        self.vehicles_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['white']};
                gridline-color: {COLORS['gray']};
                selection-background-color: {COLORS['primary']};
                selection-color: {COLORS['white']};
                border-radius: 15px;
            }}
            QHeaderView::section {{
                background-color: {COLORS['primary']};
                color: {COLORS['white']};
                padding: 10px;
                border: none;
                font-weight: bold;
            }}
            QTableWidget::item:alternate {{
                background-color: #ecf0f1;
            }}
        """)
        vehicles_layout.addWidget(self.vehicles_table)
        
        self.table_timer = QTimer(self)
        self.table_timer.timeout.connect(self.update_vehicles_table)
        self.table_timer.start(5000)
        
        tabs.addTab(violations_tab, "Нарушения")
        tabs.addTab(vehicles_tab, "Транспортные средства")
        
        right_panel.setMaximumWidth(500)
        right_layout.addWidget(tabs)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([900, 500])  # Фиксированные пропорции
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {COLORS['dark']};
                color: {COLORS['white']};
                font-size: 14px;
                border-top: 2px solid {COLORS['accent']};
            }}
        """)
        self.setStatusBar(self.statusBar)
        
        self.status_label = QLabel("Статус: Готово")
        self.statusBar.addWidget(self.status_label)
        
        self.violation_count = 0
    
    def setup_database(self):
        try:
            self.conn = sqlite3.connect('violations.db', check_same_thread=False)
            self.cursor = self.conn.cursor()
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS prapiska (
                    Plaka TEXT PRIMARY KEY,
                    Sahip TEXT,
                    Telefon TEXT,
                    YetkiliSuruculer TEXT,
                    PrapiskaTarihi TEXT
                )
            """)
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER,
                    timestamp TEXT,
                    violation_type TEXT,
                    plate TEXT,
                    owner TEXT,
                    phone TEXT,
                    vehicle_path TEXT,
                    plate_path TEXT
                )
            """)
            
            # Очистка таблицы prapiska при каждом запуске
            self.cursor.execute("DELETE FROM prapiska")
            self.conn.commit()
            
            try:
                data = pd.read_csv('dataset.csv')
                if not data.empty:
                    for _, row in data.iterrows():
                        phone = str(row['Telefon']).replace('+', '')
                        self.cursor.execute("""
                            INSERT OR REPLACE INTO prapiska (Plaka, Sahip, Telefon, YetkiliSuruculer, PrapiskaTarihi)
                            VALUES (?, ?, ?, ?, ?)
                        """, (row['Plaka'], row['Sahip'], phone, row['YetkiliSuruculer'], row['PrapiskaTarihi']))
                    self.conn.commit()
                    self.update_status("dataset.csv импортирован в SQLite и обновлен")
                    print("Содержимое dataset.csv:")
                    print(data)
                else:
                    self.update_status("dataset.csv пуст или недействителен, таблица prapiska осталась пустой")
                    print("dataset.csv пуст или недействителен")
            except FileNotFoundError:
                self.update_status("dataset.csv не найден, таблица prapiska останется пустой")
                print("dataset.csv не найден")
            except Exception as e:
                self.update_status(f"Ошибка чтения dataset.csv: {e}, таблица prapiska останется пустой")
                print(f"Ошибка чтения dataset.csv: {e}")
                
        except Exception as e:
            self.update_status(f"Ошибка базы данных: {e}")
            print(f"Ошибка базы данных: {e}")
    
    def save_violation_image(self, frame, path):
        try:
            if frame.size > 0:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                cv2.imwrite(path, frame)
                print(f"Изображение нарушения сохранено: {path}")
            else:
                print(f"Недопустимое изображение: {path}")
        except Exception as e:
            self.update_status(f"Ошибка сохранения изображения: {e}")
            print(f"Ошибка сохранения изображения: {e}")
    
    def handle_db_query(self, query, params, query_type):
        try:
            if query_type == "select":
                print(f"Запрос к базе данных: {query}, Параметры: {params}")
                self.cursor.execute(query, params)
                result = self.cursor.fetchone()
                print(f"Результат запроса к базе данных: {result}")
                
                if result and params[0]:
                    plate, owner, phone, authorized_drivers, prapiska_date = result
                    prapiska_date = datetime.datetime.strptime(prapiska_date, "%Y-%m-%d")
                    
                    status = 'valid' if prapiska_date >= datetime.datetime.now() else 'expired'
                    self.processing_thread.processed_plates[plate].update({
                        'status': status,
                        'owner': owner
                    })
                    print(f"Статус номера обновлен: {plate}, Статус: {status}, Владелец: {owner}")
                    
                    if prapiska_date < datetime.datetime.now():
                        # Если срок прописки истек
                        violation_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        track_id = int(time.time() % 1000)
                        violation_image_path = f"violations/violation_{track_id}_{violation_time.replace(':', '-')}.jpg"
                        
                        try:
                            message = (
                                f"Номер: {plate}\n"
                                f"Нарушение: Срок прописки истек\n"
                                f"Дата: {violation_time}"
                            )
                            pywhatkit.sendwhatmsg_instantly(
                                f"+{phone}",
                                message,
                                wait_time=10,
                                tab_close=True
                            )
                            print(f"Сообщение WhatsApp отправлено: {phone}")
                            self.update_status(f"Сообщение WhatsApp отправлено: {plate}")
                        except Exception as e:
                            self.update_status(f"Ошибка отправки сообщения WhatsApp: {e}. Требуется ручная отправка.")
                            print(f"Ошибка отправки сообщения WhatsApp: {e}")
                        
                        violation_info = {
                            'track_id': track_id,
                            'timestamp': violation_time,
                            'violation_type': "Срок прописки истек",
                            'plate': plate,
                            'owner': owner,
                            'phone': phone,
                            'vehicle_path': violation_image_path,
                            'plate_path': ""
                        }
                        self.processing_thread.violation_detected.emit(violation_info)
                        print(f"Сигнал о нарушении отправлен: {plate} - Срок прописки истек")
                    else:
                        # Если прописка действительна, проверяем авторизованного водителя
                        last_person = self.processing_thread.last_detected_person
                        print(f"Проверка авторизованного водителя - Обнаруженный человек: {last_person}")
                        if last_person and last_person != "Неизвестный":
                            authorized_drivers_list = authorized_drivers.split(",") if authorized_drivers else []
                            authorized_drivers_list = [driver.strip() for driver in authorized_drivers_list]
                            print(f"Авторизованные водители: {authorized_drivers_list}")
                            
                            if last_person in authorized_drivers_list:
                                self.update_status(f"{last_person} имеет разрешение на использование номера {plate}")
                                print(f"{last_person} имеет разрешение на использование номера {plate}")
                            else:
                                violation_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                track_id = int(time.time() % 1000)
                                violation_image_path = f"violations/violation_{track_id}_{violation_time.replace(':', '-')}.jpg"
                                
                                try:
                                    message = (
                                        f"Номер: {plate}\n"
                                        f"Нарушение: Неавторизованный водитель ({last_person})\n"
                                        f"Дата: {violation_time}"
                                    )
                                    pywhatkit.sendwhatmsg_instantly(
                                        f"+{phone}",
                                        message,
                                        wait_time=10,
                                        tab_close=True
                                    )
                                    print(f"Сообщение WhatsApp отправлено: {phone}")
                                    self.update_status(f"Сообщение WhatsApp отправлено: {plate} - Неавторизованный водитель")
                                except Exception as e:
                                    self.update_status(f"Ошибка отправки сообщения WhatsApp: {e}. Требуется ручная отправка.")
                                    print(f"Ошибка отправки сообщения WhatsApp: {e}")
                                
                                violation_info = {
                                    'track_id': track_id,
                                    'timestamp': violation_time,
                                    'violation_type': f"Неавторизованный водитель ({last_person})",
                                    'plate': plate,
                                    'owner': owner,
                                    'phone': phone,
                                    'vehicle_path': violation_image_path,
                                    'plate_path': ""
                                }
                                self.processing_thread.violation_detected.emit(violation_info)
                                print(f"Сигнал о нарушении отправлен: {plate} - Неавторизованный водитель")
                        else:
                            self.update_status("Обнаруженный человек неизвестен, проверка разрешения не выполнена")
                            print("Обнаруженный человек неизвестен, проверка разрешения не выполнена")
                else:
                    print(f"Номер не найден в базе данных: {params[0]}")
                    self.update_status(f"Номер не найден в базе данных: {params[0]}")
            elif query_type == "insert":
                self.cursor.execute(query, params)
                self.conn.commit()
                print(f"Добавлено в базу данных: {params}")
        except Exception as e:
            self.update_status(f"Ошибка запроса к базе данных: {e}")
            print(f"Ошибка запроса к базе данных: {e}")
    
    def start_camera(self):
        source = 0  # Веб-камера по умолчанию
        if self.processing_thread.set_camera(source):
            if not self.processing_thread.isRunning():
                self.processing_thread.start()
            self.update_status("Камера запущена")
        else:
            self.update_status("Не удалось запустить камеру! Проверьте вашу веб-камеру.")
    
    def select_video(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать видеофайл", "", "Видео (*.mp4 *.avi *.mov);;Все файлы (*.*)"
        )
        if video_path:
            if self.processing_thread.set_camera(video_path):
                if not self.processing_thread.isRunning():
                    self.processing_thread.start()
                self.update_status(f"Видео запущено: {os.path.basename(video_path)}")
            else:
                self.update_status("Не удалось запустить видео! Проверьте ваш файл.")
    
    def stop_camera(self):
        if self.processing_thread.isRunning():
            self.processing_thread.stop()
        self.update_status("Камера/видео остановлены")
        self.camera_frame.setPixmap(QPixmap())
    
    def update_frame(self, q_img):
        try:
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(780, 580, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_frame.setPixmap(scaled_pixmap)
        except Exception as e:
            self.update_status(f"Ошибка обновления кадра: {e}")
            print(f"Ошибка обновления кадра: {e}")
    
    def update_status(self, message):
        try:
            self.status_label.setText(f"Статус: {message}")
            print(f"Статус: {message}")
        except AttributeError:
            print(f"Ошибка обновления статуса: status_label не определен - {message}")
    
    def update_datetime(self):
        self.date_time_label.setText(datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
    
    def update_vehicles_table(self):
        try:
            self.vehicles_table.setRowCount(0)
            unique_plates = {}
            for plate, data in self.processing_thread.processed_plates.items():
                if data['last_processed']:
                    unique_plates[plate] = data
            
            sorted_plates = sorted(unique_plates.items(), key=lambda x: x[1]['last_processed'], reverse=True)
            
            for plate, data in sorted_plates:
                row_idx = self.vehicles_table.rowCount()
                self.vehicles_table.insertRow(row_idx)
                
                self.vehicles_table.setItem(row_idx, 0, QTableWidgetItem(plate))
                
                status_text = "Неизвестно"
                status_color = COLORS['gray']
                if data['status'] == 'valid':
                    status_text = "Действителен"
                    status_color = COLORS['success']
                elif data['status'] == 'expired':
                    status_text = "Истек"
                    status_color = COLORS['danger']
                status_item = QTableWidgetItem(status_text)
                status_item.setForeground(QColor(status_color))
                self.vehicles_table.setItem(row_idx, 1, status_item)
                
                owner = data.get('owner', "")
                self.vehicles_table.setItem(row_idx, 2, QTableWidgetItem(owner))
                
                last_time = "Неизвестно"
                if data['last_processed']:
                    last_time = data['last_processed'].strftime("%H:%M:%S")
                self.vehicles_table.setItem(row_idx, 3, QTableWidgetItem(last_time))
        except Exception as e:
            self.update_status(f"Ошибка обновления таблицы: {e}")
            print(f"Ошибка обновления таблицы: {e}")
    
    def add_violation(self, violation_info):
        try:
            vehicle_path = violation_info.get('vehicle_path', '')
            print(f"add_violation вызван: {violation_info}")

            # Проверка vehicle_path: если файл отсутствует, оставляем vehicle_path пустым
            if vehicle_path and not os.path.exists(vehicle_path):
                print(f"Изображение нарушения отсутствует: {vehicle_path}, vehicle_path сбрасывается")
                violation_info['vehicle_path'] = ""  # Если файла нет, оставляем пустым, чтобы избежать ошибки

            # Сохранение в базу данных
            self.cursor.execute("""
                INSERT INTO violations (track_id, timestamp, violation_type, plate, owner, phone, vehicle_path, plate_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                violation_info['track_id'],
                violation_info['timestamp'],
                violation_info['violation_type'],
                violation_info['plate'],
                violation_info['owner'],
                violation_info['phone'],
                violation_info['vehicle_path'],
                violation_info['plate_path']
            ))
            self.conn.commit()
            print(f"Нарушение сохранено в базу данных: {violation_info['plate']}")

            # Добавление карточки нарушения в UI
            violation_card = QFrame()
            violation_card.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLORS['light']};
                    border-radius: 15px;
                    padding: 15px;
                    margin: 10px;
                }}
            """)
            
            card_layout = QVBoxLayout(violation_card)
            card_layout.setContentsMargins(10, 10, 10, 10)
            
            header_layout = QHBoxLayout()
            violation_title = QLabel(violation_info['violation_type'])
            violation_title.setFont(QFont("Arial", 14, QFont.Bold))
            violation_title.setStyleSheet(f"color: {COLORS['danger']};")
            header_layout.addWidget(violation_title)
            
            time_label = QLabel(violation_info['timestamp'])
            time_label.setStyleSheet(f"color: {COLORS['gray']}; font-size: 12px;")
            header_layout.addWidget(time_label, alignment=Qt.AlignRight)
            
            card_layout.addLayout(header_layout)
            
            plate_label = QLabel(f"Номер: {violation_info['plate']} | Владелец: {violation_info['owner']}")
            plate_label.setStyleSheet(f"color: {COLORS['black']}; font-size: 13px; font-weight: bold;")
            card_layout.addWidget(plate_label)
            
            # Добавление карточки нарушения в violations_layout
            print(f"Количество карточек до добавления в violations_layout: {self.violations_layout.count()}")
            self.violations_layout.insertWidget(0, violation_card)
            print(f"Количество карточек после добавления в violations_layout: {self.violations_layout.count()}")
            print(f"Карточка нарушения добавлена: {violation_info['plate']}")
            
            self.violation_count += 1
            self.violation_counter.setText(f"Всего: {self.violation_count}")
            self.update_status(f"Нарушение записано: {violation_info['plate']}")
        except Exception as e:
            self.update_status(f"Ошибка добавления карточки нарушения: {e}")
            print(f"Ошибка добавления карточки нарушения: {e}")
    
    def clear_violations(self):
        confirm = QMessageBox.question(
            self, "Очистить нарушения", 
            "Все карточки нарушений будут удалены. Вы уверены?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            while self.violations_layout.count():
                item = self.violations_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                    
            self.violation_count = 0
            self.violation_counter.setText(f"Всего: {self.violation_count}")
            self.update_status("Карточки нарушений очищены")
    
    def closeEvent(self, event):
        print("Приложение закрывается. Причина и состояние закрытия:")
        try:
            print(f"Последнее сообщение о состоянии: {self.status_label.text()}")
        except AttributeError:
            print("Последнее сообщение о состоянии: status_label не определен")
        print("Состояние файлов:")
        print(f"dataset.csv существует: {'Да' if os.path.exists('dataset.csv') else 'Нет'}")
        print(f"Папка faces/ существует: {'Да' if os.path.exists('faces') else 'Нет'}")
        print(f"logo.png существует: {'Да' if os.path.exists('logo.png') else 'Нет'}")
        
        if self.processing_thread.isRunning():
            self.processing_thread.stop()
        
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        
        self.datetime_timer.stop()
        self.table_timer.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, QColor(33, 33, 33))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.Text, QColor(33, 33, 33))
    palette.setColor(QPalette.Button, QColor(255, 255, 255))
    palette.setColor(QPalette.ButtonText, QColor(33, 33, 33))
    palette.setColor(QPalette.Highlight, QColor(26, 115, 232))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())