import cv2
import numpy as np
import pickle
import os
import sys
import multiprocessing
import pyttsx3
from tensorflow.keras.models import load_model
from modules.detector_manos import Detector_Manos
from modules.frame_saver import FrameSaver
import time

sys.stdout.reconfigure(encoding='utf-8')

def motor_voz(cola_voz):
    motor = pyttsx3.init()
    motor.setProperty('rate', 500)  # Aumentar la velocidad de habla
    motor.setProperty('volume', 1.0)
    while True:
        try:
            # Usamos get_nowait para no bloquear si no hay mensajes
            text = cola_voz.get_nowait()
        except:
            continue  # Si no hay mensajes, continuamos el ciclo
        if text is None:
            break
        motor.stop()  # Detener cualquier discurso en curso
        motor.say(text)
        motor.runAndWait()

class SignLanguageRecognizer:
    def __init__(self):
        self.modo = 'reconocimiento_estatico'
        self.modelo = None
        self.le = None 
        self.scaler = None
        self.detector = Detector_Manos(max_num_hands=2, detection_confidence=0.5)
        self.saver = FrameSaver()
        self.capturing_sequence = False
        self.sequence_frames = []
        self.sequence_length = 30
        self.current_label = None
        self.frame_count = 1
        self.last_class = None
        self.cola_voz = multiprocessing.Queue()
        self.speech_process = multiprocessing.Process(target=motor_voz, args=(self.cola_voz,))
        self.speech_process.start()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.load_model_and_encoder(self.modo)
        print(f"Modo actual: {self.modo}")
        self.recognized_words = []
        self.max_palabras = 1

    def load_model_and_encoder(self, mode):
        model_path = ''
        encoder_path = ''
        scaler_path = ''
        if mode == 'reconocimiento_estatico':
            model_path = 'models/modelo_senas_estaticas.h5'
            encoder_path = 'models/label_encoder_estaticas.pkl'
            scaler_path = 'models/scaler_estaticas.pkl'
        elif mode == 'reconocimiento_dinamico':
            model_path = 'models/modelo_senas_dinamicas.h5'
            encoder_path = 'models/label_encoder_dinamicas.pkl'
            scaler_path = 'models/scaler_dinamicas.pkl'
        else:
            self.modelo = None
            self.le = None
            self.scaler = None
            return

        if os.path.exists(model_path):
            self.modelo = load_model(model_path)
            print(f"Modelo cargado desde {model_path}")
        else:
            print(f"Modelo no encontrado en {model_path}")
            self.modelo = None

        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as file:
                self.le = pickle.load(file)
            print(f"LabelEncoder cargado desde {encoder_path}")
        else:
            print(f"LabelEncoder no encontrado en {encoder_path}")
            self.le = None

        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as file:
                self.scaler = pickle.load(file)
            print(f"Scaler cargado desde {scaler_path}")
        else:
            print(f"Scaler no encontrado en {scaler_path}")
            self.scaler = None

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        frame, all_hands = self.detector.encontrar_manos(frame, draw=True)
        key = cv2.waitKey(1) & 0xFF

        if all_hands:
            if self.modo.startswith('reconocimiento'):
                if self.modelo is None or self.le is None or self.scaler is None:
                    cv2.putText(frame, 'Modelo, Encoder o Scaler no cargado', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    self.handle_recognition(frame, all_hands)
            elif self.modo == 'captura_estatica':
                self.handle_static_capture(frame, key, all_hands)
            elif self.modo == 'captura_dinamica':
                self.handle_dynamic_capture(frame, key, all_hands)
        else:
            if self.modo.startswith('captura'):
                cv2.putText(frame, 'No se detecta mano', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            self.last_class = None

        cv2.putText(frame, f'Modo (m): {self.modo}', (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Mostrar palabras reconocidas en la pantalla
        y_offset = 60
        for word in self.recognized_words:
            cv2.putText(frame, word, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 30

        cv2.imshow('Lenguaje de Señales', frame)
        self.handle_key_press(key)

    def handle_recognition(self, frame, all_hands):
        if self.modo == 'reconocimiento_estatico':
            for hand_landmarks in all_hands:
                class_name = self.predict_static(hand_landmarks)
        elif self.modo == 'reconocimiento_dinamico':
            class_name = self.predict_dynamic(frame, all_hands)

    def predict_static(self, hand_landmarks):
        landmarks = np.array(hand_landmarks).flatten().reshape(1, -1)
        landmarks_scaled = self.scaler.transform(landmarks)
        y_pred = self.modelo.predict(landmarks_scaled)
        class_id = np.argmax(y_pred)
        class_name = self.le.inverse_transform([class_id])[0]

        if class_name != self.last_class:
            self.cola_voz.put(class_name)
            self.recognized_words.append(class_name)
            if len(self.recognized_words) > self.max_palabras:
                self.recognized_words.pop(0)
            self.last_class = class_name

        return class_name

    def predict_dynamic(self, frame, all_hands):
         # Inicializar listas para las landmarks de cada mano
        hand1_landmarks = np.zeros(63)  # Suponiendo 21 puntos * 3 coordenadas
        hand2_landmarks = np.zeros(63)
    
        if len(all_hands) >= 1:
             hand1_landmarks = np.array(all_hands[0]).flatten()
        if len(all_hands) >= 2:
            hand2_landmarks = np.array(all_hands[1]).flatten()
    
        combined_landmarks = np.concatenate([hand1_landmarks, hand2_landmarks])
        self.sequence_frames.append(combined_landmarks)
    
        cv2.putText(frame, f'Capturando secuencia: {len(self.sequence_frames)}/{self.sequence_length}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        if len(self.sequence_frames) == self.sequence_length:
            sequence = np.array(self.sequence_frames)
            sequence_flat = sequence.flatten().reshape(1, -1)
            sequence_scaled = self.scaler.transform(sequence_flat)
            sequence_scaled = sequence_scaled.reshape(1, self.sequence_length, -1)
            y_pred = self.modelo.predict(sequence_scaled)
            class_id = np.argmax(y_pred)
            class_name = self.le.inverse_transform([class_id])[0]

            if class_name != self.last_class:
                self.cola_voz.put(class_name)
                self.recognized_words.append(class_name)
                if len(self.recognized_words) > self.max_palabras:
                    self.recognized_words.pop(0)
                self.last_class = class_name

            self.sequence_frames = []
            return class_name
        return None

    def handle_static_capture(self, frame, key, all_hands):
        if all_hands:
            cv2.putText(frame, 'Mano detectada - Presiona "g" para guardar imagen', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            if key == ord('g'):
                self.saver.save_image(frame, self.current_label, self.frame_count)
                print(f"Imagen estática guardada: {self.current_label}_{self.frame_count}.png")
                self.frame_count += 1
        else:
            cv2.putText(frame, 'No se detecta mano', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    def handle_dynamic_capture(self, frame, key, all_hands):
        if not self.capturing_sequence:
            cv2.putText(frame, 'Manos detectadas - Presiona "c" para iniciar captura', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            if key == ord('c'):
                self.capturing_sequence = True
                self.sequence_frames = []
                print("Iniciando captura de secuencia de seña dinámica...")
        else:
            num_landmarks_per_hand = 21 * 3  # 21 puntos clave por mano, cada uno con x, y, z
            hand1_landmarks = np.zeros(num_landmarks_per_hand)
            hand2_landmarks = np.zeros(num_landmarks_per_hand)

            if len(all_hands) >= 1:
                hand1_landmarks = np.array(all_hands[0]).flatten()
            if len(all_hands) >= 2:
                hand2_landmarks = np.array(all_hands[1]).flatten()

            # Asegurarse de que las landmarks tengan el tamaño correcto
            if hand1_landmarks.shape[0] != num_landmarks_per_hand:
                hand1_landmarks = np.zeros(num_landmarks_per_hand)
            if hand2_landmarks.shape[0] != num_landmarks_per_hand:
                hand2_landmarks = np.zeros(num_landmarks_per_hand)

            combined_landmarks = np.concatenate([hand1_landmarks, hand2_landmarks])

            self.sequence_frames.append(combined_landmarks)
            cv2.putText(frame, f'Capturando frame {len(self.sequence_frames)}/{self.sequence_length}',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            if len(self.sequence_frames) == self.sequence_length:
                sequence = np.array(self.sequence_frames)
                self.saver.save_sequence(sequence, self.current_label, self.frame_count)
                self.frame_count += 1
                self.capturing_sequence = False
                print(f"Secuencia de seña dinámica guardada: {self.current_label}_{self.frame_count}.npy")


    def handle_key_press(self, key):
        if key == ord('m'):
            self.switch_mode()
        elif key == ord('q'):
            self.limpiar()
            exit()

    def switch_mode(self):
        self.last_class = None
        self.sequence_frames = []
        self.recognized_words = []

        if self.modo == 'reconocimiento_estatico':
            self.modo = 'captura_estatica'
            self.current_label = 'W'
            self.frame_count = 1
            print(f"Cambiado a modo: {self.modo} (Captura Estática)")
        elif self.modo == 'captura_estatica':
            self.modo = 'reconocimiento_dinamico'
            self.load_model_and_encoder(self.modo)
            print(f"Cambiado a modo: {self.modo} (Reconocimiento Dinámica)")
        elif self.modo == 'reconocimiento_dinamico':
            self.modo = 'captura_dinamica'
            self.current_label = 'Hermosa'
            self.frame_count =1
            self.capturing_sequence = False
            print(f"Cambiado a modo: {self.modo} (Captura Dinámica)")
        elif self.modo == 'captura_dinamica':
            self.modo = 'reconocimiento_estatico'
            self.load_model_and_encoder(self.modo)
            print(f"Cambiado a modo: {self.modo} (Reconocimiento Estática)")

    def limpiar(self):
        self.cola_voz.put(None)
        self.speech_process.join()
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("No se pudo acceder a la cámara.")
                break
            self.process_frame(frame)
        self.limpiar()

if __name__ == '__main__':
    recognizer = SignLanguageRecognizer()
    recognizer.run()
