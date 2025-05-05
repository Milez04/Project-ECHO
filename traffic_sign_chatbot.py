import sys
import google.generativeai as genai
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLineEdit, QPushButton, QTextEdit, QLabel, QFileDialog, QStatusBar)
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QSize, QBuffer
from PIL import Image
import os
import base64
from io import BytesIO

# Gemini API anahtarını buraya yapıştır (güvenlik için .env kullanmak daha iyi)
GEMINI_API_KEY = "AIzaSyBzLQ0gOSIZsK_p1L-ux1_IINemi6K6J1E"  # Сюда вставьте свой API-ключ
genai.configure(api_key=GEMINI_API_KEY)

# Gemini modelini başlat (Gemini 1.5 Flash kullanacağız)
model = genai.GenerativeModel('gemini-1.5-flash')

class TrafficSignChatbot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система правил дорожного движения и дорожных знаков 🚗")
        self.setGeometry(100, 100, 800, 600)

        # Ana widget ve layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(15)

        # Başlık ve logo alanı
        self.header_layout = QHBoxLayout()
        
        # Logo
        self.logo_label = QLabel()
        logo_path = "logo.png"  # Логотип указан, вы добавите этот файл
        logo_pixmap = QPixmap(logo_path) if os.path.exists(logo_path) else QPixmap(32, 32)
        if logo_pixmap.isNull():
            logo_pixmap.fill(Qt.black)  # Если логотипа нет, черный placeholder
        self.logo_label.setPixmap(logo_pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.header_layout.addWidget(self.logo_label)

        # Başlık
        self.title_label = QLabel("Система правил дорожного движения и дорожных знаков")
        self.title_label.setFont(QFont("Inter", 20, QFont.Bold))
        self.title_label.setStyleSheet("color: #fff;")

        self.title_label.setAlignment(Qt.AlignCenter)
        self.header_layout.addWidget(self.title_label, stretch=1)

        self.main_layout.addLayout(self.header_layout)

        # Sohbet alanı
        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setStyleSheet("""
            QTextEdit {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #252535, stop:1 #2A2A3E);
                color: #E5E7EB;
                border: 1px solid #37374A;
                border-radius: 12px;
                padding: 12px;
                font-size: 14px;
            }
        """)
        self.main_layout.addWidget(self.chat_area)

        # Giriş alanı, fotoğraf yükleme butonu ve gönder butonu
        self.input_layout = QHBoxLayout()
        self.input_layout.setSpacing(10)

        # Fotoğraf yükleme butonu (sol tarafta, PNG ile)
        self.upload_button = QPushButton()
        upload_icon = QIcon("upload.png")  # PNG для загрузки фото, вы добавите
        self.upload_button.setIcon(upload_icon)
        self.upload_button.setIconSize(QSize(32, 32))  # Установка размера PNG
        self.upload_button.setFixedSize(40, 40)
        self.upload_button.clicked.connect(self.upload_image)
        self.input_layout.addWidget(self.upload_button)

        # Metin giriş alanı
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Напишите ваш вопрос (например: Что означают знаки скорости?)")
        self.text_input.returnPressed.connect(self.send_text_query)  # Отправка по Enter
        self.input_layout.addWidget(self.text_input)

        # Gönder butonu (sağ tarafta, PNG ile)
        self.send_button = QPushButton()
        send_icon = QIcon("send.png")  # PNG для отправки, вы добавите
        self.send_button.setIcon(send_icon)
        self.send_button.setIconSize(QSize(32, 32))  # Установка размера PNG
        self.send_button.setFixedSize(40, 40)
        self.send_button.clicked.connect(self.send_text_query)
        self.input_layout.addWidget(self.send_button)

        self.main_layout.addLayout(self.input_layout)

        # Durum çubuğu
        self.statusBar = QStatusBar()
        self.statusBar.setObjectName("statusLabel")
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Готово")

        # Stil şeması
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1A1A2E, stop:1 #16213E);
                font-family: 'Inter', sans-serif;
                color: #E5E7EB;
            }

            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #5B86E5, stop:0.5 #36D1DC, stop:1 #5B86E5);
                color: #FFFFFF;
                border-radius: 20px;
                padding: 14px 20px;
                font-size: 15px;
                font-weight: 600;
                border: 2px solid rgba(91, 134, 229, 0.5);
                text-align: center;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            }

            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #36D1DC, stop:0.5 #5B86E5, stop:1 #36D1DC);
                box-shadow: 0 6px 14px rgba(54, 209, 220, 0.3);
                border-color: #36D1DC;
            }

            QPushButton:pressed {
                background: #4A90E2;
                box-shadow: inset 0 3px 6px rgba(0,0,0,0.3);
            }

            QLineEdit {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #252535, stop:1 #2A2A3E);
                color: #E5E7EB;
                border: 1px solid #37374A;
                border-radius: 12px;
                padding: 12px;
                font-size: 14px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            QLineEdit:focus {
                border: 2px solid #5B86E5;
                background: #2A2A3E;
                box-shadow: 0 4px 8px rgba(91, 134, 229, 0.2);
            }

            QLabel#statusLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #252535, stop:1 #2A2A3E);
                color: #D1D5DB;
                font-size: 14px;
                padding: 14px;
                border-radius: 12px;
                border: 1px solid #37374A;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }
        """)

        # Fotoğraf değişkeni
        self.current_image = None

        # İlk mesaj (hoş geldin mesajı)
        self.add_message("Система", "Привет! Вы можете задать вопросы о правилах дорожного движения или загрузить фотографию дорожного знака для получения информации.", is_system=True )

    def add_message(self, sender, message, is_system=False, is_user=False, image_path=None):
        """Sohbet alanına bir mesaj veya fotoğraf ekler."""
        current_text = self.chat_area.toHtml()
        
        if image_path:
            # Fotoğrafı base64 formatına çevirerek HTML'de göster
            pixmap = QPixmap(image_path).scaled(200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            buffer = QBuffer()
            buffer.open(QBuffer.WriteOnly)
            pixmap.save(buffer, "PNG")
            buffer.close()
            image_data = base64.b64encode(buffer.data()).decode('utf-8')
            image_html = f"""
            <div style='background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #5B86E5, stop:1 #36D1DC); 
                         color: #FFFFFF; border-radius: 10px; padding: 10px; margin: 5px 0; 
                         font-size: 14px; text-align: right; margin-left: 20%;'>
                <b>{sender}:</b><br>
                <img src='data:image/png;base64,{image_data}' alt='Загруженное фото'/>
            </div>
            """
            self.chat_area.setHtml(current_text + image_html)
        elif is_system:
            message_html = f"""
            <div style='background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #37374A, stop:1 #4B4B5E); 
                         color: #E5E7EB; border-radius: 10px; padding: 10px; margin: 5px 0; 
                         font-size: 14px; text-align: center;'>
                <b>{sender}:</b> {message}
            </div>
            """
            self.chat_area.setHtml(current_text + message_html)
        elif is_user:
            message_html = f"""
            <div style='background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #5B86E5, stop:1 #36D1DC); 
                         color: #FFFFFF; border-radius: 10px; padding: 10px; margin: 5px 0; 
                         font-size: 14px; text-align: right; margin-left: 20%;'>
                <b>{sender}:</b> {message}
            </div>
            """
            self.chat_area.setHtml(current_text + message_html)
        else:
            message_html = f"""
            <div style='background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #37374A, stop:1 #4B4B5E); 
                         color: #E5E7EB; border-radius: 10px; padding: 10px; margin: 5px 0; 
                         font-size: 14px; text-align: left; margin-right: 20%;'>
                <b>{sender}:</b> {message}
            </div>
            """
            self.chat_area.setHtml(current_text + message_html)
        
        self.chat_area.verticalScrollBar().setValue(self.chat_area.verticalScrollBar().maximum())

    def send_text_query(self):
        query = self.text_input.text().strip()
        if not query:
            self.statusBar.showMessage("Пожалуйста, напишите вопрос!")
            return

        # Kullanıcı mesajını ekle
        self.add_message("Вы", query, is_user=True)
        self.text_input.clear()

        self.statusBar.showMessage("Ожидание ответа...")
        try:
            response = model.generate_content(query)
            self.add_message("Gemini", response.text)
            self.statusBar.showMessage("Ответ получен")
        except Exception as e:
            self.add_message("Система", f"Произошла ошибка: {e}", is_system=True)
            self.statusBar.showMessage(f"Ошибка: {e}")

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить фото знака", "", "Изображения (*.jpg *.jpeg *.png)"
        )
        if file_path:
            self.current_image = Image.open(file_path)
            self.add_message("Вы", "", is_user=True, image_path=file_path)
            self.statusBar.showMessage("Фото загружено")

            # Otomatik olarak analiz et
            self.analyze_image()
        else:
            self.statusBar.showMessage("Фото не загружено")

    def analyze_image(self):
        if self.current_image is None:
            self.statusBar.showMessage("Пожалуйста, загрузите фото!")
            return

        self.statusBar.showMessage("Анализ знака...")
        try:
            prompt = "Что означает этот знак? Пожалуйста, объясните."
            response = model.generate_content([prompt, self.current_image])
            self.add_message("Gemini", response.text)
            self.statusBar.showMessage("Анализ завершен")
        except Exception as e:
            self.add_message("Система", f"Произошла ошибка: {e}", is_system=True)
            self.statusBar.showMessage(f"Ошибка: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrafficSignChatbot()
    window.show()
    sys.exit(app.exec_())