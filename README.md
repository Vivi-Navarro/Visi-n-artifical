# Sistema de Detección de Trampa - Visión Clásica

Este proyecto es un sistema académico diseñado para detectar posibles trampas en exámenes virtuales utilizando únicamente técnicas clásicas de visión por computadora.

## Restricciones Técnicas
- Sin Machine Learning / Deep Learning.
- Sin modelos pre-entrenados modernos (YOLO, MediaPipe, etc.).
- Uso exclusivo de OpenCV, Haar Cascades y procesamiento de imágenes.

## Funcionalidades
1. Detección de mirada fuera de pantalla.
2. Detección de giros de cabeza excesivos.
3. Alerta por ausencia de rostro o rostro parcialmente fuera de cuadro.
4. Sistema Multimedia: Reproducción sincronizada de video y audio de alerta al detectar una infracción persistente.
5. Alarma sonora multiplataforma (Windows, Linux, macOS).

## Estructura del Repositorio
```
/
├── data/
│   ├── cascades/          # Archivos XML de clasificadores Haar
│   ├── sounds/            # Archivos de audio de alerta (alerta.mp3)
│   └── videos/            # Archivos de video de alerta (alerta.mp4)
├── docs/
│   └── architecture.md    # Documentación técnica y arquitectura
├── src/
│   ├── main.py            # Punto de entrada del programa
│   ├── capture.py         # Gestión de cámara
│   ├── detection.py       # Algoritmos de detección facial y ocular
│   ├── analysis.py        # Lógica de detección de infracciones
│   └── alerts.py          # Gestión de alertas visuales, sonoras y multimedia
├── requirements.txt       # Dependencias de Python
└── README.md              # Información general
```

## Requisitos
- Python 3.x
- OpenCV (opencv-python)
- NumPy
- Playsound (para reproducción de audio multiplataforma)

## Instalación
1. Clonar el repositorio.
2. Instalar dependencias: `pip install -r requirements.txt`.
3. Descargar los archivos Haar Cascade:
   - [haarcascade_frontalface_default.xml](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml)
   - [haarcascade_eye.xml](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml)
   - Colócalos en `data/cascades/`.
4. Preparar Alertas Multimedia:
   - Coloca tu video en `data/videos/alerta.mp4`.
   - Coloca tu audio en `data/sounds/alerta.mp3`.
5. Ejecutar: `python src/main.py`.

## Uso del Sistema
- Al detectar una infracción persistente (2 segundos), se abrirá una ventana de alerta a la derecha con el video y el audio sincronizados.
- La ventana de alerta se puede cerrar manualmente presionando la tecla **'q'**.
- Una vez cerrada la alerta, el sistema continuará monitoreando y podrá activarse nuevamente si se detecta otra falta.
