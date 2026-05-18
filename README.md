# Detector de Trampa en Exámenes - Visión Artificial 👁️

Un programa que usa tu cámara web para detectar si estás haciendo trampa en un examen virtual.  
**No usa inteligencia artificial**, solo procesamiento clásico de imágenes con OpenCV.

## ¿Qué hace?

- Detecta si miras para otro lado (izquierda, derecha, arriba, abajo)
- Detecta si giras mucho la cabeza
- Detecta si te tapas los ojos o los cierras mucho tiempo
- Detecta si te sales del encuadre de la cámara
- Detecta si no hay nadie frente a la cámara
- Te da 5 oportunidades (strikes) antes de cerrar el examen
- Suena una alarma y pone un marco rojo cuando detecta algo

## ¿Qué necesitas?

- **Python 3.10** (o superior)
- **OpenCV** y **NumPy** (se instalan con pip)
- **Windows** (para el sonido de alarma usa winsound, que viene incluido)

## ¿Cómo lo instalo?

1. Clona o descarga este repositorio
2. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```
3. Descarga estos 2 archivos y ponlos en la carpeta `data/cascades/`:
   - [haarcascade_frontalface_default.xml](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml)
   - [haarcascade_eye.xml](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml)
4. Ejecuta:
   ```
   python src/main.py
   ```

## ¿Cómo funciona?

### Paso 1: Calibración (5 segundos)
Al iniciar, el programa te pide que mires al centro de la pantalla con los ojos abiertos durante 5 segundos. En ese tiempo aprende dónde están tus pupilas "normalmente". Eso se llama el **baseline**.

### Paso 2: Monitoreo
Después de calibrar, el programa empieza a vigilar en tiempo real:
1. Busca tu cara usando Haar Cascades (unos archivos XML de OpenCV)
2. Dentro de tu cara, busca tus ojos
3. Dentro de cada ojo, busca la pupila con 3 métodos diferentes
4. Compara la posición de la pupila con el baseline
5. Si miras fuera por más de 0.6 segundos → strike
6. Con 5 strikes → se cierra el examen

## Teclas

| Tecla | ¿Qué hace? |
|-------|-------------|
| `q`   | Salir del programa |
| `d`   | Activar/desactivar el modo debug (muestra info técnica abajo) |
| `r`   | Recalibrar (vuelve a los 5 segundos de calibración) |

## Archivos del proyecto

```
├── data/cascades/          ← Archivos XML de OpenCV para detectar caras y ojos
├── docs/architecture.md    ← Explicación de cómo funciona por dentro
├── src/
│   ├── main.py             ← Archivo principal, aquí se ejecuta todo
│   ├── capture.py          ← Maneja la cámara web
│   ├── detection.py        ← Detecta caras, ojos y pupilas
│   ├── analysis.py         ← Decide si hay trampa o no
│   └── alerts.py           ← Maneja las alertas (sonido y visual)
├── requirements.txt        ← Las librerías que necesitas instalar
└── README.md               ← Este archivo que estás leyendo
```

## Restricciones del proyecto

Este proyecto se hizo como tarea académica con estas reglas:
- ❌ Sin Machine Learning / Deep Learning
- ❌ Sin modelos pre-entrenados modernos (YOLO, MediaPipe, etc.)
- ✅ Solo OpenCV clásico + Haar Cascades + procesamiento de imágenes
