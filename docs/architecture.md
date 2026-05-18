# Arquitectura del Sistema de Detección de Trampa

Este sistema utiliza técnicas de **Visión por Computadora clásica** para monitorear el comportamiento de un usuario durante un examen virtual. Se basa exclusivamente en procesamiento de imágenes, geometría y heurísticas temporales — **sin modelos de aprendizaje profundo, redes neuronales ni dependencias modernas como MediaPipe o dlib**.

## Restricciones Técnicas
- Sin Machine Learning / Deep Learning.
- Sin modelos pre-entrenados modernos (YOLO, MediaPipe, etc.).
- Uso exclusivo de OpenCV (incluidos Haar Cascades clásicos) y procesamiento de imágenes.

---

## Componentes Principales

### 1. Módulo de Captura (`capture.py`)
Gestiona la entrada de video desde la cámara web utilizando OpenCV.
- Inicialización de `cv2.VideoCapture` con fallback automático a un segundo dispositivo.
- Lectura de frames en tiempo real.

### 2. Módulo de Preprocesamiento (integrado en `main.py`)
Prepara la imagen antes de la detección.
- Redimensionamiento a resolución de procesamiento (640×480).
- Conversión a escala de grises.
- Ecualización adaptativa de histograma con **CLAHE** (`createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`) para normalizar la iluminación.

### 3. Módulo de Detección (`detection.py`)

#### 3.1 Detección de Rostro
- `cv2.CascadeClassifier` con `haarcascade_frontalface_default.xml`.
- Parámetros: `scaleFactor=1.1`, `minNeighbors=5`, `minSize=(80,80)`.

#### 3.2 Detección de Ojos
- `cv2.CascadeClassifier` con `haarcascade_eye.xml`.
- Restricción a la banda media del rostro (18 %–55 % de la altura) — evita confundir cejas o nariz.
- **Filtro anti-cejas** (`_looks_like_eye`): valida que la región tenga un punto oscuro central (iris/pupila) significativamente más oscuro que los bordes superior e inferior. Una ceja, al ser una franja uniforme, no pasa el filtro.

#### 3.3 Detección de Pupila — cascada de 3 métodos
La pupila se busca con métodos progresivamente más permisivos hasta encontrar un candidato válido:

1. **HoughCircles** sobre el ROI del ojo con ecualización de histograma y filtro gaussiano.
2. **Pipeline estilo GazeTracking** (atribución MIT — ver al final):
   `bilateralFilter` → `erode×3` → `threshold` a 3 niveles (40, 50, 60) → `findContours` con validación de circularidad.
3. **Fallback de contornos adaptativos** (`adaptiveThreshold` + morfología).

Cada candidato se valida por:
- Posición dentro de una **banda vertical central** (20 %–80 % del alto del ojo) — evita los párpados.
- **Oscuridad máxima** (la pupila debe medir < 90 en escala 0–255).
- **Circularidad** del contorno.
- Cuando hay pupila previa, se prefiere el candidato **más cercano** a la posición anterior (continuidad temporal) en lugar del más oscuro absoluto, lo cual evita confusiones con sombras de párpado.

#### 3.4 Detección de Ojo Cerrado
Se calcula la **varianza vertical del ROI del ojo**. Un ojo abierto presenta un cambio fuerte de intensidad (esclera → iris → pupila), mientras que un ojo cerrado es una franja uniformemente oscura. Si la varianza cae por debajo del umbral, se reporta pupila = `None`.

### 4. Suavizado Temporal (parte de `detection.py`)

Capa que reduce el ruido frame a frame de las detecciones de Haar y la pupila.

- **EMA (Exponential Moving Average)** sobre los bounding boxes de cada ojo. Si el nuevo cuadro está cerca del anterior, se promedia; si saltó más allá del tamaño del ojo, se acepta como movimiento real.
- **EMA + persistencia 0.2 s** en el centro de la pupila. Si HoughCircles falla un frame y existe una pupila reciente, se mantiene brevemente para evitar parpadeo en pantalla.

### 5. Calibración Inicial (parte de `detection.py` + `main.py`)

Fase de **5 segundos** al inicio del programa:
- Se muestra un countdown grande pidiendo al usuario mirar al centro de la pantalla con los ojos abiertos.
- Solo se acumulan muestras cuando **ambas pupilas están detectadas simultáneamente**.
- Al finalizar se calcula la **mediana** de los ratios (x/w, y/h) de cada ojo y se guarda como **baseline personal**.
- Si no se acumularon al menos 8 muestras válidas, el sistema usa umbrales fijos clásicos como respaldo.
- Tecla `r` permite **recalibrar en caliente** sin reiniciar.

### 6. Módulo de Análisis (`analysis.py`)

Aplica lógica geométrica y temporal sobre las detecciones.

- **Dirección de la mirada:** desplazamiento de la pupila respecto al baseline calibrado, con tolerancia configurable (`0.13`).
- **Giro de cabeza:** se evalúa la relación de aspecto (`alto/ancho`) del bounding box del rostro y su tamaño relativo al frame.
- **Ausencia de rostro:** suavizada con un historial de los últimos 8 frames; se reporta solo si ≥ 70 % no detectan rostro.
- **Salida parcial:** se comprueba si el bbox del rostro toca los márgenes (4 %) del encuadre.
- **Confirmación temporal:** una violación solo se acepta como real si se sostiene durante **N frames consecutivos** (por defecto 4) y supera el umbral de tiempo configurado.

### 7. Módulo de Alertas (`alerts.py`)

Ejecuta las acciones de advertencia.

- **Alerta Visual:** marco rojo en el borde del frame + texto con la razón.
- **Alarma Sonora no bloqueante:** `winsound.Beep` lanzado en un *thread* separado con un *lock* para evitar solapamiento de pitidos. Evita congelar el video durante el beep.

### 8. Control de Strikes (parte de `main.py`)

- **Cooldown de 3 segundos** entre strikes acumulados — evita que una sola violación sostenida sume múltiples strikes en ráfaga.
- Al llegar a 5 strikes se muestra la pantalla "EXAMEN CERRADO" y el programa termina.

---

## Flujo de Datos

```
1. Frame capturado
2. Preprocesado (resize → grayscale → CLAHE)
3. Detección de rostro (Haar)
4. Detección de ojos (Haar + filtro anti-cejas)
5. Detección de pupila (cascada de 3 métodos + validaciones)
6. Suavizado temporal (EMA bbox + EMA pupila + persistencia)
7. ¿Fase de calibración?
   ├── Sí → acumular muestras y mostrar countdown
   └── No → continuar al paso 8
8. Análisis (gaze, giro, ausencia, salida parcial, ojo cerrado)
9. Confirmación temporal (N frames consecutivos)
10. ¿Violación confirmada y cooldown vencido?
    ├── Sí → incrementar strike + alerta visual + alarma sonora async
    └── No → continuar
11. ¿Strikes ≥ 5? → cerrar examen
12. Mostrar frame en pantalla
13. Loop
```

---

## Atribuciones

El método `_pupil_gazetracking_style` en `src/detection.py` está adaptado de:

- **[antoinelame/GazeTracking](https://github.com/antoinelame/GazeTracking)** — Licencia MIT, Copyright © 2019 Antoine Lamé.

Se reutiliza **únicamente** el algoritmo de pupila puramente clásico (`bilateralFilter` + `erode` + `threshold` + `findContours`). NO se importan los componentes que dependen de `dlib` ni del modelo pre-entrenado `shape_predictor_68_face_landmarks.dat`. La adaptación respeta las restricciones técnicas listadas al inicio.
