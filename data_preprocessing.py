import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.2  # Valor ajustado para mayor sensibilidad
        )

    def process_images(self):
        X = []  # Datos de entrada
        y = []  # Etiquetas
        labels = os.listdir(self.data_dir)
        print(f"Etiquetas encontradas: {labels}")
        total_images = 0
        total_processed = 0
        for label in labels:
            label_dir = os.path.join(self.data_dir, label)
            if not os.path.isdir(label_dir):
                continue
            print(f"Procesando etiqueta: {label}")
            image_count = 0
            processed_count = 0
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Imagen no encontrada o no se pudo leer: {img_path}")
                    continue
                landmarks = self.extract_landmarks(image)
                if landmarks is not None:
                    X.append(landmarks)
                    y.append(label)
                    processed_count += 1
                else:
                    print(f"No se detectó mano en la imagen: {img_path}. Eliminando...")
                    try:
                        os.remove(img_path)
                        print(f"Imagen eliminada: {img_path}")
                    except Exception as e:
                        print(f"Error al eliminar {img_path}: {e}")
                image_count += 1
            print(f"Total de imágenes en {label}: {image_count}")
            print(f"Imágenes procesadas exitosamente en {label}: {processed_count}")
            total_images += image_count
            total_processed += processed_count
        print(f"Total de imágenes: {total_images}")
        print(f"Total de imágenes procesadas: {total_processed}")
        return np.array(X), np.array(y)

    def extract_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return landmarks
        else:
            return None  # Si no se detecta mano, devuelve None

if __name__ == '__main__':
    # Preprocesamiento de datos
    preprocessor = DataPreprocessor(data_dir='data')
    X, y = preprocessor.process_images()

    print(f"Total de muestras procesadas: {X.shape[0]}")
    print("Distribución de clases:")
    unique_labels, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Etiqueta {label}: {count} muestras")

    # Codificación de etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Guardar el codificador de etiquetas
    if not os.path.exists('models'):
        os.makedirs('models')
    with open('models/label_encoder.pkl', 'wb') as file:
        pickle.dump(le, file)

    # División de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.1, stratify=y_encoded, random_state=42)

    # Verificar la distribución en los conjuntos de entrenamiento y prueba
    print("Distribución en el conjunto de entrenamiento:")
    unique_labels_train, counts_train = np.unique(y_train, return_counts=True)
    for label, count in zip(unique_labels_train, counts_train):
        print(f"Etiqueta {le.inverse_transform([label])[0]}: {count} muestras")

    print("Distribución en el conjunto de prueba:")
    unique_labels_test, counts_test = np.unique(y_test, return_counts=True)
    for label, count in zip(unique_labels_test, counts_test):
        print(f"Etiqueta {le.inverse_transform([label])[0]}: {count} muestras")

    # Guardar los conjuntos de datos procesados (opcional)
    np.save('models/X_train.npy', X_train)
    np.save('models/X_test.npy', X_test)
    np.save('models/y_train.npy', y_train)
    np.save('models/y_test.npy', y_test)

    print('Preprocesamiento completado.')
