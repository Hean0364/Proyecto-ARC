# main.py

import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import sys
# import pyttsx3  # Librería para texto a voz (si la usas)
from tensorflow.keras.models import load_model
from modules.hand_detector import HandDetector
from modules.frame_saver import FrameSaver
import time

sys.stdout.reconfigure(encoding='utf-8')

def main():
    # Inicializar el motor de texto a voz (si lo utilizas)
    # engine = pyttsx3.init()
    # engine.setProperty('rate', 150)
    # engine.setProperty('volume', 1.0)

    # Estado inicial del modo
    mode = 'reconocimiento_estatico'  

    # Variable para almacenar la última clase detectada
    last_class = None

    # Cargar los modelos y codificadores según el modo
    model = None
    le = None
    scaler = None  # Añadido para el scaler

    def cargar_modelo_y_encoder(modo):
        nonlocal model, le, scaler
        if modo == 'reconocimiento_estatico':
            model_path = 'models/modelo_senas_estaticas.h5'
            encoder_path = 'models/label_encoder_estaticas.pkl'
            scaler_path = 'models/scaler_estaticas.pkl'
        elif modo == 'reconocimiento_dinamico':
            model_path = 'models/modelo_senas_dinamicas.h5'
            encoder_path = 'models/label_encoder_dinamicas.pkl'
            scaler_path = 'models/scaler_dinamicas.pkl'
        else:
            model = None
            le = None
            scaler = None
            return

        if os.path.exists(model_path):
            model = load_model(model_path)
            print(f"Modelo cargado desde {model_path}")
        else:
            print(f"Modelo no encontrado en {model_path}")
            model = None

        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as file:
                le = pickle.load(file)
            print(f"LabelEncoder cargado desde {encoder_path}")
        else:
            print(f"LabelEncoder no encontrado en {encoder_path}")
            le = None

        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as file:
                scaler = pickle.load(file)
            print(f"Scaler cargado desde {scaler_path}")
        else:
            print(f"Scaler no encontrado en {scaler_path}")
            scaler = None

    # Cargar el modelo y encoder para el modo inicial
    cargar_modelo_y_encoder(mode)

    # Inicializar el detector de manos
    detector = HandDetector(max_num_hands=1, detection_confidence=0.5)

    # Inicializar el guardador de frames
    saver = FrameSaver()

    # Variables para captura dinámica
    capturing_sequence = False
    sequence_frames = []
    sequence_length = 30  # Ajusta este valor según tus necesidades
    current_label = None
    frame_count = 1

    # Configurar la captura de video
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Modo actual: {mode}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la cámara.")
            break

        frame = cv2.flip(frame, 1)
        frame, all_hands = detector.find_hands(frame, draw=True)

        key = cv2.waitKey(1) & 0xFF  # Capturar la tecla presionada

        if all_hands:
            hand_landmarks = all_hands[0]

            if mode.startswith('reconocimiento'):
                if model is None or le is None or scaler is None:
                    cv2.putText(frame, 'Modelo, Encoder o Scaler no cargado', (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    if mode == 'reconocimiento_estatico':
                        # Realizar predicción inmediatamente
                        landmarks = np.array(hand_landmarks).flatten().reshape(1, -1)
                        # Aplicar el mismo preprocesamiento (escalado)
                        landmarks_scaled = scaler.transform(landmarks)
                        y_pred = model.predict(landmarks_scaled)
                        class_id = np.argmax(y_pred)
                        class_name = le.inverse_transform([class_id])[0]

                        # Mostrar la predicción en la imagen
                        cv2.putText(frame, f'Letra: {class_name}', (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Convertir la letra detectada a voz si es una nueva detección
                        # if class_name != last_class:
                        #     engine.say(class_name)
                        #     engine.runAndWait()
                        #     last_class = class_name

                    elif mode == 'reconocimiento_dinamico':
                        # Capturar los landmarks y agregarlos a la secuencia
                        landmarks = np.array(hand_landmarks).flatten()
                        sequence_frames.append(landmarks)

                        # Mostrar progreso
                        cv2.putText(frame, f'Capturando secuencia: {len(sequence_frames)}/{sequence_length}', 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

                        if len(sequence_frames) == sequence_length:
                            # Convertir la lista de frames a un numpy array
                            sequence = np.array(sequence_frames)  # Forma: (sequence_length, n_features)

                            # Aplanar la secuencia
                            sequence_flat = sequence.flatten().reshape(1, -1)  # Forma: (1, sequence_length * n_features)

                            # Aplicar el mismo preprocesamiento
                            sequence_scaled = scaler.transform(sequence_flat)

                            # Restaurar la forma original para el modelo
                            sequence_scaled = sequence_scaled.reshape(1, sequence_length, -1)  # Forma: (1, sequence_length, n_features)

                            # Realizar la predicción
                            y_pred = model.predict(sequence_scaled)
                            class_id = np.argmax(y_pred)
                            class_name = le.inverse_transform([class_id])[0]

                            # Mostrar la predicción
                            cv2.putText(frame, f'Seña: {class_name}', (10, 60), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                            # Convertir la seña detectada a voz si es una nueva detección
                            # if class_name != last_class:
                            #     engine.say(class_name)
                            #     engine.runAndWait()
                            #     last_class = class_name

                            # Reiniciar la secuencia
                            sequence_frames = []

            elif mode == 'captura_estatica':
                if all_hands:
                    cv2.putText(frame, 'Mano detectada - Presiona "g" para guardar imagen', (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'No se detecta mano', (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            elif mode == 'captura_dinamica':
                if not capturing_sequence:
                    cv2.putText(frame, 'Mano detectada - Presiona "c" para iniciar captura', (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                    if key == ord('c'):
                        capturing_sequence = True
                        sequence_frames = []
                        print("Iniciando captura de secuencia de seña dinámica...")

                if capturing_sequence:
                    landmarks = np.array(hand_landmarks).flatten()
                    sequence_frames.append(landmarks)

                    cv2.putText(frame, f'Capturando frame {len(sequence_frames)}/{sequence_length}', 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                    if len(sequence_frames) == sequence_length:
                        sequence = np.array(sequence_frames)
                        saver.save_sequence(sequence, current_label, frame_count)
                        frame_count += 1
                        capturing_sequence = False
                        print(f"Secuencia de seña dinámica guardada: {current_label}_{frame_count}.npy")

        else:
            if mode.startswith('captura'):
                cv2.putText(frame, 'No se detecta mano', (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f'Modo (m): {mode}', (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Lenguaje de Señales', frame)

        if key == ord('m'):
            if mode == 'reconocimiento_estatico':
                mode = 'captura_estatica'
                current_label = 'W'
                frame_count = 1
                print(f"Cambiado a modo: {mode} (Captura Estática)")
            elif mode == 'captura_estatica':
                mode = 'reconocimiento_dinamico'
                cargar_modelo_y_encoder(mode)
                print(f"Cambiado a modo: {mode} (Reconocimiento Dinámica)")
            elif mode == 'reconocimiento_dinamico':
                mode = 'captura_dinamica'
                current_label = 'Z'  # Cambia 'Z' por la etiqueta que desees capturar
                frame_count = 1
                capturing_sequence = False  # Asegúrate de reiniciar esta variable
                print(f"Cambiado a modo: {mode} (Captura Dinámica)")
            elif mode == 'captura_dinamica':
                mode = 'reconocimiento_estatico'
                cargar_modelo_y_encoder(mode)
                last_class = None
                sequence_frames = []
                print(f"Cambiado a modo: {mode} (Reconocimiento Estática)")

        if mode == 'captura_estatica':
            if key == ord('g') and all_hands:
                saver.save_image(frame, current_label, frame_count)
                print(f"Imagen estática guardada: {current_label}_{frame_count}.png")
                frame_count += 1

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
