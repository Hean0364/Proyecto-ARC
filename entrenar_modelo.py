"""
import numpy as np
from model_trainer_static import entrenar_modelo
import pickle
import sys
sys.stdout.reconfigure(encoding='utf-8')




if __name__ == '__main__':
    # Cargar los datos preprocesados
    X_train = np.load('models/X_train.npy')
    X_test = np.load('models/X_test.npy')
    y_train = np.load('models/y_train.npy')
    y_test = np.load('models/y_test.npy')

    # Cargar el codificador de etiquetas
    with open('models/label_encoder.pkl', 'rb') as file:
        le = pickle.load(file)

    num_classes = len(le.classes_)

    # Entrenar el modelo
    model = entrenar_modelo(X_train, y_train, X_test, y_test, num_classes)

    # Evaluar el modelo
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Precisión en el conjunto de prueba: {accuracy * 100:.2f}%')

    # Guardar el modelo entrenado
    model.save('models/modelo_senas.h5')
    print('Modelo guardado exitosamente.')

from sklearn.metrics import confusion_matrix, classification_report

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generar la matriz de confusión
cm = confusion_matrix(y_test, y_pred_classes)
print('Matriz de Confusión:')
print(cm)

# Generar el reporte de clasificación
report = classification_report(y_test, y_pred_classes, target_names=le.classes_)
print('Reporte de Clasificación:')
print(report)
"""