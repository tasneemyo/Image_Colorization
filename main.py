import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from gui import Ui_MainWindow
from PIL import Image
import numpy as np
from notebooks.unet import UNet
import torch
import torchvision.transforms as T
# from colorize import colorize   

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.apply_styles()

        self.image_path = None
        self.weights_path="Weights/best_unet_colorization.pth"
        # Connect buttons
        self.ui.upload.clicked.connect(self.upload_image)
        self.ui.colorize.clicked.connect(self.process_image)
        self.model=UNet()
        self.model.load_state_dict(torch.load("Weights/best_unet_colorization.pth", map_location="cpu"))
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

        # Load input image
        img = Image.open(self.image_path).convert("L")   # grayscale

        # Preprocess
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0)  # (1, 1, H, W)

        # Model inference
        with torch.no_grad():
            output = self.model(img_tensor)       # (1, 2, H, W)

        # Convert logits → probabilities
        output = torch.softmax(output, dim=1)

        # Take predicted channel
        pred = torch.argmax(output, dim=1).squeeze(0)    # (H, W)

        # Convert to displayable image
        pred_np = (pred.cpu().numpy() * 255).astype(np.uint8)
        output_img = Image.fromarray(pred_np)

        # Save temporary result
        output_img.save("temp_output.png")
        pixmap = QPixmap("temp_output.png")

        # Show result
        self.ui.colorized_img.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
