
import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mediapipe as mp  # Importación correcta de MediaPipe

class DataPreprocessorStatic:
    def __init__(self, data_dir='data/estaticas'):
        self.data_dir = data_dir
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.2  # Mayor sensibilidad
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

def preprocesar_datos(X, y):
    """
    Preprocesa los datos estáticos: normalización y codificación de etiquetas.

    Args:
        X (np.ndarray): Matriz de secuencias de landmarks.
        y (np.ndarray): Arreglo de etiquetas correspondientes a cada secuencia.

    Returns:
        X_train (np.ndarray): Conjunto de entrenamiento preprocesado.
        X_test (np.ndarray): Conjunto de prueba preprocesado.
        y_train (np.ndarray): Etiquetas de entrenamiento codificadas.
        y_test (np.ndarray): Etiquetas de prueba codificadas.
        le (LabelEncoder): Objeto LabelEncoder ajustado.
        scaler (StandardScaler): Objeto StandardScaler ajustado.
    """
    # Aplanar las secuencias para aplicar la normalización
    num_samples, features = X.shape
    X_flat = X.reshape((num_samples, features))

    # Normalizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    # Codificación de etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # División de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.1, stratify=y_encoded, random_state=42)

    print("Preprocesamiento completado para datos estáticos.")
    return X_train, X_test, y_train, y_test, le, scaler

def guardar_datos(X_train, X_test, y_train, y_test, le, scaler, output_dir='models'):
    """
    Guarda los conjuntos de datos preprocesados y los objetos de codificación.

    Args:
        X_train (np.ndarray): Conjunto de entrenamiento preprocesado.
        X_test (np.ndarray): Conjunto de prueba preprocesado.
        y_train (np.ndarray): Etiquetas de entrenamiento codificadas.
        y_test (np.ndarray): Etiquetas de prueba codificadas.
        le (LabelEncoder): Objeto LabelEncoder ajustado.
        scaler (StandardScaler): Objeto StandardScaler ajustado.
        output_dir (str): Directorio donde se guardarán los archivos.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Guardar los conjuntos de datos
    np.save(os.path.join(output_dir, 'X_train_estaticas.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test_estaticas.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train_estaticas.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test_estaticas.npy'), y_test)

    # Guardar el codificador de etiquetas y el scaler
    with open(os.path.join(output_dir, 'label_encoder_estaticas.pkl'), 'wb') as file:
        pickle.dump(le, file)

    with open(os.path.join(output_dir, 'scaler_estaticas.pkl'), 'wb') as file:
        pickle.dump(scaler, file)

    print(f"Datos preprocesados guardados en la carpeta '{output_dir}'.")

if __name__ == '__main__':
    # Preprocesamiento de datos estáticos
    preprocessor = DataPreprocessorStatic(data_dir='data/estaticas')
    X, y = preprocessor.process_images()

    print(f"Total de muestras procesadas: {X.shape[0]}")
    print("Distribución de clases:")
    unique_labels, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Etiqueta {label}: {count} muestras")

    # Preprocesar los datos
    X_train, X_test, y_train, y_test, le, scaler = preprocesar_datos(X, y)

    # Guardar los datos preprocesados y objetos de codificación
    guardar_datos(X_train, X_test, y_train, y_test, le, scaler, output_dir='models')

    print('Preprocesamiento de imágenes estáticas completado.')
