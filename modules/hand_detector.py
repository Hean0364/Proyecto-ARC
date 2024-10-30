import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, max_num_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.max_num_hands = max_num_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Inicializar MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence)
        self.mp_drawing = mp.solutions.drawing_utils

    def find_hands(self, image, draw=True):
        # Convertir la imagen a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        all_hands = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    # Dibujar los puntos clave y conexiones
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                # Obtener coordenadas de los puntos clave
                hand = []
                for lm in hand_landmarks.landmark:
                    hand.append([lm.x, lm.y, lm.z])
                all_hands.append(hand)
        return image, all_hands
