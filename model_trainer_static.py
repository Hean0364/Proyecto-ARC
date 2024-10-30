# model_trainer_static.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pickle
import os

def entrenar_modelo_simple(X_train, y_train, X_test, y_test, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
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

    # Cargar los conjuntos de datos preprocesados
    X_train = np.load('models/X_train_estaticas.npy')
    X_test = np.load('models/X_test_estaticas.npy')
    y_train = np.load('models/y_train_estaticas.npy')
    y_test = np.load('models/y_test_estaticas.npy')

    # Cargar el LabelEncoder
    with open('models/label_encoder_estaticas.pkl', 'rb') as file:
        le = pickle.load(file)

    # Número de clases
    num_classes = len(le.classes_)

    # Entrenar el modelo
    model = entrenar_modelo_simple(X_train, y_train, X_test, y_test, num_classes)

    # Guardar el modelo entrenado
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/modelo_senas_estaticas.h5')
    print("Modelo estático entrenado y guardado exitosamente.")
