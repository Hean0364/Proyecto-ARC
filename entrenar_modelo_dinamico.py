
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import sys
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.model_selection import train_test_split
import pickle
import os

def cargar_datos_dynamic(data_dir='data/dinamicas', label_encoder_path='models/label_encoder_dinamicas.pkl'):
    X = []
    y = []
    labels = os.listdir(data_dir)
    print(f"Etiquetas encontradas para dinámicas: {labels}")

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for file_name in os.listdir(label_dir):
            if file_name.endswith('.npy'):
                file_path = os.path.join(label_dir, file_name)
                sequence = np.load(file_path)
                X.append(sequence)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"Total de secuencias cargadas: {X.shape[0]}")

    # Cargar el codificador de etiquetas
    with open(label_encoder_path, 'rb') as file:
        le = pickle.load(file)

    y_encoded = le.transform(y)

    return X, y_encoded, le

def entrenar_modelo_lstm(X, y, num_classes):
    # Redimensionar los datos para LSTM (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = Sequential([
        LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.5),
        LSTM(64),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, batch_size=32,
              validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Evaluar el modelo en el conjunto de prueba
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Precisión en el conjunto de prueba: {accuracy*100:.2f}%')

    return model

if __name__ == '__main__':
    # Cargar y preprocesar los datos dinámicos
    X, y, le = cargar_datos_dynamic()

    # Número de clases
    num_classes = len(np.unique(y))

    # Entrenar el modelo LSTM
    model = entrenar_modelo_lstm(X, y, num_classes)

    # Guardar el modelo entrenado
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/modelo_senas_dinamicas.h5')
    print("Modelo entrenado y guardado exitosamente.")
