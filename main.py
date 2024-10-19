import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import sys
import pyttsx3  # Librería para texto a voz
from tensorflow.keras.models import load_model
from modules.hand_detector import HandDetector
from modules.frame_saver import FrameSaver
import time

# Configurar la salida estándar para soportar UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def main():
    # Inicializar el motor de texto a voz
    engine = pyttsx3.init()
    engine.setProperty('rate', 130)  # Velocidad de la voz (opcional)
    engine.setProperty('volume', 1.0)  # Volumen de la voz (opcional)

    # Estado inicial del modo
    mode = 'reconocimiento_estatico'  # Modos: 'reconocimiento_estatico', 'captura_estatica', 'reconocimiento_dinamico', 'captura_dinamica'

    # Variable para almacenar la última clase detectada
    last_class = None

    # Cargar los modelos y codificadores según el modo
    model = None
    le = None

    def cargar_modelo_y_encoder(modo):
        nonlocal model, le
        if modo == 'reconocimiento_estatico':
            model_path = 'models/modelo_senas_estaticas.h5'
            encoder_path = 'models/label_encoder_estaticas.pkl'
        elif modo == 'reconocimiento_dinamico':
            model_path = 'models/modelo_senas_dinamicas.h5'
            encoder_path = 'models/label_encoder_dinamicas.pkl'
        else:
            model = None
            le = None
            return

        # Cargar el modelo
        if os.path.exists(model_path):
            model = load_model(model_path)
            print(f"Modelo cargado desde {model_path}")
        else:
            print(f"Modelo no encontrado en {model_path}")
            model = None

        # Cargar el LabelEncoder
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as file:
                le = pickle.load(file)
            print(f"LabelEncoder cargado desde {encoder_path}")
        else:
            print(f"LabelEncoder no encontrado en {encoder_path}")
            le = None

    # Cargar el modelo y encoder para el modo inicial
    cargar_modelo_y_encoder(mode)

    # Inicializar el detector de manos
    detector = HandDetector(max_num_hands=1, detection_confidence=0.5)

    # Inicializar el guardador de frames
    saver = FrameSaver()

    # Variables para captura dinámica
    capturing_sequence = False
    sequence_frames = []
    sequence_length = 30  # Número de frames a capturar por seña dinámica
    current_label = None
    frame_count = 1  # Contador de frames guardados

    # Variables para manejar la pausa y la frecuencia de predicciones en reconocimiento estático
    static_frame_count = 0
    static_prediction_interval = 10  # Realizar predicción cada 10 frames

    # Configurar la captura de video con una resolución más baja para mejorar el rendimiento
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Mostrar el modo inicial en la consola
    print(f"Modo actual: {mode}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la cámara.")
            break

        # Voltear la imagen horizontalmente para efecto espejo
        frame = cv2.flip(frame, 1)

        # Detectar manos y obtener los puntos clave
        frame, all_hands = detector.find_hands(frame, draw=True)

        if all_hands:
            # Tomar la primera mano detectada
            hand_landmarks = all_hands[0]

            if mode.startswith('reconocimiento'):
                if model is None or le is None:
                    cv2.putText(frame, 'Modelo o Encoder no cargado', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    if mode == 'reconocimiento_estatico':
                        # Incrementar el contador de frames
                        static_frame_count += 1

                        # Realizar predicción solo cada `static_prediction_interval` frames
                        if static_frame_count >= static_prediction_interval:
                            # Procesamiento para señas estáticas
                            landmarks = np.array(hand_landmarks).flatten().reshape(1, -1)
                            y_pred = model.predict(landmarks)
                            class_id = np.argmax(y_pred)
                            class_name = le.inverse_transform([class_id])[0]

                            # Mostrar la predicción en la imagen
                            cv2.putText(frame, f'Letra: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 255, 0), 2, cv2.LINE_AA)

                            # Convertir la letra detectada a voz si es una nueva detección
                            if class_name != last_class:
                                engine.say(class_name)
                                engine.runAndWait()
                                last_class = class_name

                            # Reiniciar el contador de frames
                            static_frame_count = 0
                        else:
                            # Mostrar un indicador de que el sistema está listo para predecir
                            cv2.putText(frame, f'Preparado para predecir en {static_prediction_interval - static_frame_count} frames', 
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

                    elif mode == 'reconocimiento_dinamico':
                        # Procesamiento para señas dinámicas
                        landmarks = np.array(hand_landmarks).flatten()
                        sequence_frames.append(landmarks)

                        if len(sequence_frames) == sequence_length:
                            sequence = np.array(sequence_frames).reshape(1, sequence_length, -1)
                            y_pred = model.predict(sequence)
                            class_id = np.argmax(y_pred)
                            class_name = le.inverse_transform([class_id])[0]

                            # Mostrar la predicción en la imagen
                            cv2.putText(frame, f'Seña: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 255, 0), 2, cv2.LINE_AA)

                            # Convertir la seña detectada a voz si es una nueva detección
                            if class_name != last_class:
                                engine.say(class_name)
                                engine.runAndWait()
                                last_class = class_name

                            # Reiniciar la captura de secuencia
                            sequence_frames = []
                        else:
                            cv2.putText(frame, f'Capturando seña dinámica: {len(sequence_frames)}/{sequence_length}', (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            elif mode.startswith('captura'):
                if mode == 'captura_estatica':
                    # Mostrar mensaje indicando que se detectó una mano
                    cv2.putText(frame, 'Mano detectada - Presiona "g" para guardar imagen', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 0), 2, cv2.LINE_AA)
                elif mode == 'captura_dinamica':
                    # Mostrar mensaje indicando que se detectó una mano
                    cv2.putText(frame, 'Mano detectada - Presiona "c" para iniciar captura de secuencia', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 0), 2, cv2.LINE_AA)

                    # Iniciar captura de secuencia al presionar 'c'
                    if not capturing_sequence and key == ord('c'):
                        capturing_sequence = True
                        sequence_frames = []
                        print("Iniciando captura de secuencia de seña dinámica...")

                    # Capturar la secuencia de frames si está capturando
                    if capturing_sequence:
                        landmarks = np.array(hand_landmarks).flatten()
                        sequence_frames.append(landmarks)

                        cv2.putText(frame, f'Capturando frame {len(sequence_frames)}/{sequence_length}', (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                        if len(sequence_frames) == sequence_length:
                            # Guardar la secuencia completa
                            sequence = np.array(sequence_frames)
                            saver.save_sequence(sequence, current_label, frame_count)
                            frame_count += 1
                            capturing_sequence = False  # Finalizar captura
                            print(f"Secuencia de seña dinámica guardada: {current_label}_{frame_count}.npy")

        else:
            # Manejar frames sin detección de manos
            if mode.startswith('reconocimiento_dinamico') and len(sequence_frames) > 0:
                # Append zeros para frames sin detección
                landmarks = np.zeros(21*3)  # Asumiendo 21 landmarks con x, y, z
                sequence_frames.append(landmarks)

                if len(sequence_frames) == sequence_length:
                    sequence = np.array(sequence_frames).reshape(1, sequence_length, -1)
                    y_pred = model.predict(sequence)
                    class_id = np.argmax(y_pred)
                    class_name = le.inverse_transform([class_id])[0]

                    # Mostrar la predicción en la imagen
                    cv2.putText(frame, f'Seña: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2, cv2.LINE_AA)

                    # Convertir la seña detectada a voz si es una nueva detección
                    if class_name != last_class:
                        engine.say(class_name)
                        engine.runAndWait()
                        last_class = class_name

                    # Reiniciar la captura de secuencia
                    sequence_frames = []
                else:
                    cv2.putText(frame, f'Capturando seña dinámica: {len(sequence_frames)}/{sequence_length}', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # Mostrar mensaje si no se detecta una mano en modos de captura
            if mode.startswith('captura'):
                cv2.putText(frame, 'No se detecta mano', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # Mostrar el modo actual en la esquina inferior izquierda de la imagen
        cv2.putText(frame, f'Modo (m): {mode}', (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Mostrar la imagen resultante
        cv2.imshow('Lenguaje de Señales', frame)

        # Capturar la tecla presionada
        key = cv2.waitKey(1) & 0xFF

        # Manejo de teclas para cambiar de modo y otras acciones
        if key == ord('m'):  # Presiona 'm' para cambiar de modo
            if mode == 'reconocimiento_estatico':
                mode = 'captura_estatica'
                current_label = 'A'  # Cambia a la seña estática que deseas capturar
                frame_count = 1  # Reiniciar el contador según sea necesario
                print(f"Cambiado a modo: {mode} (Captura Estática)")
            elif mode == 'captura_estatica':
                mode = 'reconocimiento_dinamico'
                cargar_modelo_y_encoder(mode)
                print(f"Cambiado a modo: {mode} (Reconocimiento Dinámica)")
            elif mode == 'reconocimiento_dinamico':
                mode = 'captura_dinamica'
                current_label = 'LL'  # Cambia a la seña dinámica que deseas capturar
                frame_count = 1  # Reiniciar el contador según sea necesario
                print(f"Cambiado a modo: {mode} (Captura Dinámica)")
            elif mode == 'captura_dinamica':
                mode = 'reconocimiento_estatico'
                cargar_modelo_y_encoder(mode)
                last_class = None  # Reiniciar la última clase detectada
                print(f"Cambiado a modo: {mode} (Reconocimiento Estática)")

        # Manejo de teclas específicas según el modo
        if mode == 'captura_estatica':
            if key == ord('g') and all_hands:
                # Guardar la imagen estática
                saver.save_image(frame, current_label, frame_count)
                print(f"Imagen estática guardada: {current_label}_{frame_count}.png")
                frame_count += 1
        elif mode == 'captura_dinamica':
            # La captura dinámica ya se maneja dentro del bloque de captura dinámica
            pass

        if key == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
