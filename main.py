import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
from tensorflow.keras.models import load_model
from modules.hand_detector import HandDetector
from modules.frame_saver import FrameSaver
import sys
sys.stdout.reconfigure(encoding='utf-8')

def main():
    
    mode = 'reconocimiento'  # Cambia a 'reconocimiento' para el modo de reconocimiento

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
        current_label = 'B'  # Cambia 'C' por la letra o seña que deseas capturar
        frame_count = 321  # Contador de frames guardados

    # Configurar la captura de video
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la camara.")
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
            elif mode == 'captura':
                # Mostrar mensaje indicando que se detectó una mano
                cv2.putText(frame, 'Mano detectada - Presiona "g" para guardar', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            # Mostrar mensaje si no se detecta una mano
            cv2.putText(frame, 'No se detecta mano', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        # Mostrar la imagen resultante
        cv2.imshow('Lenguaje de Senas', frame)

        # Capturar la tecla presionada
        key = cv2.waitKey(1) & 0xFF

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
