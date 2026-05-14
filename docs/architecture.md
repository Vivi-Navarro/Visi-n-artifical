# Arquitectura del Sistema de Detección de Trampa

Este sistema utiliza técnicas de Visión por Computadora clásica para monitorear el comportamiento de un usuario durante un examen virtual. Se basa exclusivamente en el procesamiento de imágenes y geometría, sin el uso de modelos de aprendizaje profundo o redes neuronales.

## Componentes Principales

### 1. Módulo de Captura (Capture Module)
Encargado de gestionar la entrada de video desde la cámara web utilizando OpenCV.
- Configuración de resolución y FPS.
- Captura de frames en tiempo real.

### 2. Módulo de Preprocesamiento (Preprocessing Module)
Prepara la imagen para mejorar la precisión de los algoritmos de detección.
- Conversión a escala de grises.
- Reducción de ruido mediante filtros gaussianos.
- Ecualización de histogramas para normalizar la iluminación.

### 3. Módulo de Detección (Detection Module)
Utiliza Clasificadores en Cascada de Haar para identificar estructuras faciales.
- **Detección de Rostro:** Identificación del rectángulo que contiene la cara.
- **Detección de Ojos:** Localización de las regiones oculares dentro del rostro.
- **Detección de Pupilas:** Aplicación de umbralización (thresholding) y búsqueda de contornos para hallar el centro del iris.

### 4. Módulo de Análisis (Analysis Module)
Aplica lógica geométrica y temporal para determinar posibles infracciones.
- **Dirección de la mirada:** Calcula el desplazamiento de la pupila respecto al centro del ojo.
- **Giro de cabeza:** Monitorea la relación de aspecto del rostro y la presencia de perfiles.
- **Ausencia/Salida parcial:** Verifica si el clasificador de rostro devuelve resultados válidos y si estos se encuentran dentro de los márgenes del encuadre.
- **Temporizador de Infracción:** Acumula frames de conducta sospechosa antes de activar la alarma.

### 5. Módulo de Alertas (Alert Module)
Ejecuta las acciones de advertencia.
- **Alerta Visual:** Superposición de texto y marcos rojos en el stream de video.
- **Alarma Sonora:** Generación de pitidos de advertencia mediante el sistema.

## Flujo de Datos
1. Frame capturado -> 2. Convertir a Grises -> 3. Detectar Rostro -> 4. Detectar Ojos -> 5. Localizar Pupilas -> 6. Evaluar Condiciones -> 7. ¿Infracción? -> 8. Activar Alerta.
