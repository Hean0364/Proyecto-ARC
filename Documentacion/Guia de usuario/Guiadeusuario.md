## Como usarlo

1. Clonarlo en tu dispositivo

git clone https://github.com/Hean0364/Proyecto-ARC

2. Establecer un entorno virual (venv)

3. Para trabajo en colaboracion con git, crear .gitignore (opcional)

## Establecer entorno virtual

En la terminal:

1.  python -m venv venv "o" python3 -m venv venv 
2. virtualenv venv  (Crea la carpeta)
3.  venv\Scripts\activate  # En Windows

source venv/bin/activate  # En Linux
4. pip install numpy opencv-python tensorflow scikit-learn mediapipe pyttsx3
5.. pip freeze > requerimientos.txt

## Para ejecutar el programa

1. Ejecutar procesar_datos_dinamicos.py
2. Ejecutar procesar_datos_estaticos.py
3. Ejecutar entrenar_modelo_dinamico.py
4. Ejectuar entrenar_modelo_estatico.py
4. Ejectuar main.py