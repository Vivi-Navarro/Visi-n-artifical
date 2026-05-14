# Sistema de Detección de Trampa - Visión Clásica

Este proyecto es un sistema académico diseñado para detectar posibles trampas en exámenes virtuales utilizando únicamente técnicas clásicas de visión por computadora.

## Restricciones Técnicas
- Sin Machine Learning / Deep Learning.
- Sin modelos pre-entrenados modernos (YOLO, MediaPipe, etc.).
- Uso exclusivo de OpenCV, Haar Cascades y procesamiento de imágenes.

## Funcionalidades
1. Detección de mirada fuera de pantalla.
2. Detección de giros de cabeza excesivos.
3. Alerta por ausencia de rostro.
4. Alerta por rostro parcialmente fuera de cuadro.
5. Alarma sonora y visual tras persistencia de infracción.

## Estructura del Repositorio
```
/
├── data/
│   └── cascades/          # Archivos XML de clasificadores Haar
├── docs/
│   └── architecture.md    # Documentación técnica y arquitectura
├── src/
│   ├── main.py            # Punto de entrada del programa
│   ├── capture.py         # Gestión de cámara
│   ├── detection.py       # Algoritmos de detección facial y ocular
│   ├── analysis.py        # Lógica de detección de infracciones
│   └── alerts.py          # Gestión de alertas visuales y sonoras
├── requirements.txt       # Dependencias de Python
└── README.md              # Información general
```

## Requisitos
- Python 3.x
- OpenCV (opencv-python)
- NumPy
- Winsound (integrado en Windows para la alarma sonora)

## Instalación
1. Clonar el repositorio.
2. Instalar dependencias: `pip install -r requirements.txt`.
3. Descargar los archivos Haar Cascade:
   - [haarcascade_frontalface_default.xml](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml)
   - [haarcascade_eye.xml](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml)
   - Colócalos en la carpeta `data/cascades/`.
4. Ejecutar: `python src/main.py`.
