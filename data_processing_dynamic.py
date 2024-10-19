# data_processing_dynamic.py

import os
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def cargar_secuencias_dinamicas(data_dir='data/dinamicas'):
    """
    Carga todas las secuencias dinámicas almacenadas en archivos .npy.

    Args:
        data_dir (str): Directorio base donde se almacenan las secuencias dinámicas.

    Returns:
        X (np.ndarray): Matriz de secuencias de landmarks.
        y (list): Lista de etiquetas correspondientes a cada secuencia.
    """
    X = []
    y = []
    labels = os.listdir(data_dir)
    print(f"Etiquetas encontradas para dinámicas: {labels}")

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        print(f"Cargando secuencias para la etiqueta: {label}")

        for file_name in os.listdir(label_dir):
            if file_name.endswith('.npy'):
                file_path = os.path.join(label_dir, file_name)
                try:
                    sequence = np.load(file_path)
                    X.append(sequence)
                    y.append(label)
                except Exception as e:
                    print(f"Error al cargar {file_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"Total de secuencias cargadas: {X.shape[0]}")
    return X, y

def preprocesar_datos(X, y):
    """
    Preprocesa los datos dinámicos: normalización y codificación de etiquetas.

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
    num_samples, timesteps, features = X.shape
    X_flat = X.reshape((num_samples, timesteps * features))

    # Normalizar los datos
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)

    # Restaurar la forma original después de la normalización
    X_scaled = X_flat_scaled.reshape((num_samples, timesteps, features))

    # Codificación de etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # División de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    print("Preprocesamiento completado para datos dinámicos.")
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
    np.save(os.path.join(output_dir, 'X_train_dinamicas.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test_dinamicas.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train_dinamicas.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test_dinamicas.npy'), y_test)

    # Guardar el codificador de etiquetas y el scaler
    with open(os.path.join(output_dir, 'label_encoder_dinamicas.pkl'), 'wb') as file:
        pickle.dump(le, file)

    with open(os.path.join(output_dir, 'scaler_dinamicas.pkl'), 'wb') as file:
        pickle.dump(scaler, file)

    print(f"Datos preprocesados guardados en la carpeta '{output_dir}'.")

if __name__ == '__main__':
    # Directorio de datos dinámicos
    data_dir = 'data/dinamicas'

    # Cargar secuencias dinámicas
    X, y = cargar_secuencias_dinamicas(data_dir)

    # Verificar si hay datos cargados
    if X.shape[0] == 0:
        print("No se encontraron secuencias dinámicas para procesar.")
        exit()

    # Preprocesar los datos
    X_train, X_test, y_train, y_test, le, scaler = preprocesar_datos(X, y)

    # Guardar los datos preprocesados y objetos de codificación
    guardar_datos(X_train, X_test, y_train, y_test, le, scaler, output_dir='models')
