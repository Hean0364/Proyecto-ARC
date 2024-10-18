import os
import cv2

class FrameSaver:
    def __init__(self, base_dir='data'):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def save_frame(self, image, label, count):
        label_dir = os.path.join(self.base_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        frame_name = os.path.join(label_dir, f'{label}_{count}.png')
        cv2.imwrite(frame_name, image)
        print(f"Imagen guardada en: {os.path.abspath(frame_name)}")
