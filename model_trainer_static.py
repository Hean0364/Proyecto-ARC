# model_trainer_static.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import sys
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.model_selection import train_test_split
import pickle
import os

def cargar_datos_static(data_dir='data/estaticas', label_encoder_path='models/label_encoder.pkl'):
    X = []
    y = []
    labels = os.listdir(data_dir)
    print(f"Etiquetas encontradas para estáticas: {labels}")

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for img_name in os.listdir(label_dir):
            if img_name.endswith('.png'):
                img_path = os.path.join(label_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Imagen no encontrada o no se pudo leer: {img_path}")
                    continue
                landmarks = extract_landmarks(image)
                if landmarks is not None:
                    X.append(landmarks)
                    y.append(label)
                else:
                    print(f"No se detectó mano en la imagen: {img_path}. Eliminando...")
                    try:
                        os.remove(img_path)
                        print(f"Imagen eliminada: {img_path}")
                    except Exception as e:
                        print(f"Error al eliminar {img_path}: {e}")
    X = np.array(X)
    y = np.array(y)

    print(f"Total de imágenes cargadas: {X.shape[0]}")

    # Cargar el codificador de etiquetas
    with open(label_encoder_path, 'rb') as file:
        le = pickle.load(file)

    y_encoded = le.transform(y)

    return X, y_encoded, le

def extract_landmarks(image):
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.2
    )
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    else:
        return None

def entrenar_modelo_simple(X_train, y_train, X_test, y_test, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(63,)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, batch_size=16,
              validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Evaluar el modelo en el conjunto de prueba
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Precisión en el conjunto de prueba: {accuracy*100:.2f}%')

    return model

if __name__ == '__main__':
    import cv2  # Importar aquí para evitar conflictos

    # Cargar y preprocesar los datos estáticos
    X, y, le = cargar_datos_static()

    # Número de clases
    num_classes = len(np.unique(y))

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42)

    # Entrenar el modelo
    model = entrenar_modelo_simple(X_train, y_train, X_test, y_test, num_classes)

    # Guardar el modelo entrenado
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/modelo_senas_estaticas.h5')
    print("Modelo estático entrenado y guardado exitosamente.")
