# modules/frame_saver.py

import os
import cv2
import numpy as np

class FrameSaver:
    def __init__(self, base_dir='data'):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def save_image(self, image, label, count):
        label_dir = os.path.join(self.base_dir, 'estaticas', label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        frame_name = os.path.join(label_dir, f'{label}_{count}.png')
        cv2.imwrite(frame_name, image)
        print(f"Imagen estática guardada en: {frame_name}")

    def save_sequence(self, sequence, label, count):
        label_dir = os.path.join(self.base_dir, 'dinamicas', label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        sequence_name = os.path.join(label_dir, f'{label}_{count}.npy')
        np.save(sequence_name, sequence)
        print(f"Secuencia dinámica guardada en: {sequence_name}")
