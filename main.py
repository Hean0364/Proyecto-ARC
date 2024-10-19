import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import sys
import pyttsx3  # Librería para texto a voz
from tensorflow.keras.models import load_model
from modules.detector_manos import HandDetector
from modules.frame_saver import FrameSaver

# Configurar la salida estándar para soportar UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def main():
    # Inicializar el motor de texto a voz
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Velocidad de la voz (opcional)
    engine.setProperty('volume', 1.0)  # Volumen de la voz (opcional)

    # Estado inicial del modo
    mode = 'reconocimiento'  # Modo inicial: 'reconocimiento' o 'captura'

    # Variable para almacenar la última clase detectada
    last_class = None

    if mode == 'reconocimiento':
        # Cargar el modelo entrenado y el codificador de etiquetas
        model = load_model('models/modelo_senas.h5')
        with open('models/label_encoder.pkl', 'rb') as file:
            le = pickle.load(file)

    # Inicializar el detector de manos
    detector = HandDetector(max_num_hands=1, detection_confidence=0.5)

    # Si estamos en modo captura, inicializar el guardador de frames
    if mode == 'captura':
        saver = FrameSaver()
        # Etiqueta de la seña actual (modificar según la seña que estás capturando)
        current_label = 'D'  # Cambia 'J' por la letra o seña que deseas capturar
        frame_count = 202  # Contador de frames guardados

    # Configurar la captura de video
    cap = cv2.VideoCapture(0)

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

            if mode == 'reconocimiento':
                # Convertir los puntos clave en un array numpy
                landmarks = np.array(hand_landmarks).flatten().reshape(1, -1)

                # Realizar la predicción
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

            elif mode == 'captura':
                # Mostrar mensaje indicando que se detectó una mano
                cv2.putText(frame, 'Mano detectada - Presiona "g" para guardar', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            # Mostrar mensaje si no se detecta una mano
            cv2.putText(frame, 'No se detecta mano', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        # Mostrar el modo actual en la esquina inferior izquierda de la imagen
        cv2.putText(frame, f'Modo: {mode}', (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Mostrar la imagen resultante
        cv2.imshow('Lenguaje de Señales', frame)

        # Capturar la tecla presionada
        key = cv2.waitKey(1) & 0xFF

        # Manejo de teclas para cambiar de modo y otras acciones
        if key == ord('m'):  # Presiona 'm' para cambiar de modo
            if mode == 'reconocimiento':
                mode = 'captura'
                # Inicializar el guardador de frames para el nuevo modo
                saver = FrameSaver()
                current_label = 'D'  # Cambia 'J' por la letra o seña que deseas capturar
                frame_count = 202  # Reiniciar o ajustar el contador según sea necesario
                print(f"Cambiado a modo: {mode}")
            else:
                mode = 'reconocimiento'
                # Cargar el modelo y el codificador nuevamente
                model = load_model('models/modelo_senas.h5')
                with open('models/label_encoder.pkl', 'rb') as file:
                    le = pickle.load(file)
                last_class = None  # Reiniciar la última clase detectada
                print(f"Cambiado a modo: {mode}")

        if mode == 'captura':
            if key == ord('g') and all_hands:
                # Guardar el frame actual
                saver.save_frame(frame, current_label, frame_count)
                print(f"Imagen guardada: {current_label}_{frame_count}.png")
                frame_count += 1
            elif key == ord('q'):
                break
        elif mode == 'reconocimiento':
            if key == ord('q'):
                break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
