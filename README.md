# Sistema de Detección de Trampa - Visión Clásica

Este proyecto es un sistema académico diseñado para detectar posibles trampas en exámenes virtuales utilizando únicamente técnicas clásicas de visión por computadora.

## Restricciones Técnicas
- Sin Machine Learning / Deep Learning.
- Sin modelos pre-entrenados modernos (YOLO, MediaPipe, etc.).
- Uso exclusivo de OpenCV, Haar Cascades y procesamiento de imágenes.

## Funcionalidades
1. Detección de mirada fuera de pantalla (left / right / up / down).
2. Detección de giros de cabeza excesivos.
3. Alerta por ausencia de rostro (con suavizado anti-falsos positivos).
4. Alerta por rostro parcialmente fuera de cuadro.
5. Detección de ojo cerrado por análisis de varianza vertical.
6. Calibración inicial de 5 segundos con baseline relativo al usuario.
7. Recalibración en caliente sin reiniciar el programa.
8. Cooldown configurable entre strikes (evita conteos en ráfaga).
9. Confirmación temporal de violaciones (N frames consecutivos).
10. Alarma sonora no bloqueante y alerta visual con marco rojo.
11. Cierre automático del examen tras acumular 5 strikes.

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
│   ├── detection.py       # Algoritmos de detección facial, ocular y de pupila
│   ├── analysis.py        # Lógica de detección de infracciones
│   └── alerts.py          # Gestión de alertas visuales y sonoras
├── requirements.txt       # Dependencias de Python
└── README.md              # Información general
```

## Requisitos
- Python 3.10 (probado) — debería funcionar en 3.8+.
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

## Cómo funciona

### Fase 1 — Calibración (5 segundos)
Al iniciar, el sistema entra en modo calibración con un countdown visible. El usuario debe **mirar al centro de la pantalla con los ojos abiertos**. Durante esos 5 segundos se acumulan muestras de la posición de la pupila en cada ojo (solo se aceptan muestras con ambos ojos detectados). Al finalizar se calcula la mediana y se guarda como **baseline personal**.

### Fase 2 — Monitoreo
Cada frame:
1. Se detecta el rostro y los ojos con Haar Cascades.
2. La posición de la pupila se obtiene con una cascada de 3 métodos: HoughCircles → pipeline estilo GazeTracking (bilateralFilter + erosión + threshold + contornos) → contornos adaptativos como fallback.
3. La pupila se suaviza temporalmente con EMA y persistencia de 0.2 s.
4. Se compara la posición actual contra el baseline calibrado.
5. Si el usuario se desvía más allá de la tolerancia (~0.13) por más de 0.6 segundos, se dispara un strike.
6. Tras 5 strikes acumulados, el examen se cierra automáticamente.

## Controles

| Tecla | Acción |
|-------|--------|
| `q`   | Salir del programa |
| `d`   | Toggle del overlay de debug (muestra dirección, confianza, violación, tiempo) |
| `r`   | Recalibrar en caliente (vuelve a la fase 1 sin reiniciar) |

## Atribuciones

El pipeline secundario de detección de pupila (`_pupil_gazetracking_style` en `src/detection.py`) está adaptado de:

- **[antoinelame/GazeTracking](https://github.com/antoinelame/GazeTracking)** — Licencia MIT, Copyright © 2019 Antoine Lamé.

Se reutiliza únicamente el algoritmo puramente clásico de pupila (`bilateralFilter` + `erode` + `threshold` + `findContours`), sin la dependencia de dlib ni de los landmarks pre-entrenados que el repo original usa. La adaptación mantiene las restricciones técnicas listadas arriba.
