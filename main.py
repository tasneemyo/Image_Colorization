import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from Gans.utils import lab_to_rgb
from Gans.val import postprocess_and_display, preprocess_image
from gui import Ui_MainWindow
from PIL import Image
import numpy as np
from Gans import model
import torch
import torchvision.transforms as T
from torchvision import transforms

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.apply_styles()

        self.image_path = None
        self.weights_path="Weights/Unet_colorization_model.pth"
        # Connect buttons
        self.ui.upload.clicked.connect(self.upload_image)
        self.ui.colorize.clicked.connect(self.process_image)
        self.model=model.MainModel()
        self.model.load_state_dict(torch.load(self.weights_path, map_location="cpu"))
        self.model.eval()
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e; /* Dark purple background */
            }

            QLabel#gray, QLabel#colorized_img {
                background-color: #2a2a40;
                border: 2px solid #6c63ff;
                border-radius: 15px;
                padding: 5px;
            }

            QPushButton {
                background-color: #6c63ff;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 12px;
                font-size: 14px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #857df2;
            }

            QPushButton:pressed {
                background-color: #5148d2;
            }

            QMenuBar {
                background-color: #2a2a40;
                color: white;
            }

            QStatusBar {
                background-color: #2a2a40;
                color: #cccccc;
            }
        """)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.ui.gray.setPixmap(pixmap)

    def process_image(self):
        if not self.image_path:
            return
        self.model.net_G.eval()  # Set the model to evaluation mode
        
        

        # Load input image
        transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
        gray = Image.open(self.image_path).convert("RGB")
        # gray=cv2.imread(self.image_path)
        L_tensor = preprocess_image(gray, 256)  # Preprocess the image
        with torch.no_grad():
            fake_ab = self.model.net_G(L_tensor) 
         # Predict ab channels
        L = L_tensor.cpu()
        ab = fake_ab.cpu()
        fake_rgb = lab_to_rgb(L, ab)[0]
        # print(fake_rgb)
        fake_rgb = transforms.ToPILImage()(fake_rgb)
        
        
        fake_rgb.save("temp_output.png")
        pixmap = QPixmap("temp_output.png")

        # Show result
        self.ui.colorized_img.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
