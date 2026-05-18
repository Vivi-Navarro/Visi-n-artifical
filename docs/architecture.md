# ¿Cómo funciona el sistema por dentro? 🔍

Explicación sencilla de cada parte del programa.

---

## 1. La Cámara (`capture.py`)

- Abre tu cámara web usando OpenCV
- Si la cámara principal (índice 0) no funciona, prueba con otra (índice 1)
- Cada vez que el programa lo pide, captura una "foto" (frame)
- Al terminar, libera la cámara para que otras apps la puedan usar

---

## 2. Preparación de la imagen (dentro de `main.py`)

Antes de analizar cada frame, lo preparamos:
- Lo hacemos más chiquito (640x480) para que sea más rápido de procesar
- Lo convertimos a blanco y negro (escala de grises)
- Le aplicamos **CLAHE** para mejorar la iluminación (que no se vea muy oscuro ni muy claro)

---

## 3. El Detector (`detection.py`)

Este es el archivo más grande y el cerebro del sistema. Hace varias cosas:

### 3.1 Buscar la cara
- Usa un archivo XML llamado Haar Cascade que sabe reconocer caras
- Le dice al programa: "aquí hay una cara, en estas coordenadas"

### 3.2 Buscar los ojos
- Dentro de la zona de la cara, busca los ojos con otro Haar Cascade
- Solo acepta ojos que estén en la zona correcta (no muy arriba = frente, no muy abajo = nariz)
- Tiene un filtro anti-cejas: verifica que el centro sea más oscuro que los bordes (así funciona un ojo real)

### 3.3 Buscar la pupila
Busca la pupila con 3 métodos, uno tras otro, hasta que alguno funcione:

1. **Círculos de Hough**: busca formas circulares en la zona del ojo
2. **Pipeline GazeTracking**: usa filtros + erosión + contornos (inspirado en un proyecto open source)
3. **Contornos adaptativos**: último recurso, más permisivo

Cada candidato a pupila se valida así:
- ¿Está en la zona correcta del ojo? (no en los párpados)
- ¿Es lo suficientemente oscuro? (la pupila es oscura)
- ¿Es redondo? (la pupila es circular)
- ¿Está cerca de donde estaba antes? (para que no salte a cualquier lado)

### 3.4 Detectar ojo cerrado
- Si la imagen del ojo se ve muy uniforme (poca variación de claro a oscuro), probablemente está cerrado
- En un ojo abierto hay cambio fuerte: blanco (esclera) → oscuro (pupila) → blanco

### 3.5 Suavizado
Para que las detecciones no tiemblen de un frame a otro:
- Los recuadros de los ojos se suavizan con un promedio entre el frame actual y el anterior
- La posición de la pupila también se suaviza
- Si la pupila no se detecta en un frame pero había una reciente, la mantenemos 0.2 segundos

---

## 4. La Calibración (dentro de `detection.py` + `main.py`)

Al iniciar el programa:
- Tienes 5 segundos para mirar al centro de la pantalla
- El sistema guarda dónde están tus pupilas cuando miras al centro
- Eso se llama el **baseline** (tu posición "normal")
- Solo cuenta muestras cuando detecta AMBOS ojos a la vez
- Si no juntó suficientes muestras, usa unos valores genéricos como respaldo
- Puedes recalibrar en cualquier momento con la tecla `r`

---

## 5. El Analizador (`analysis.py`)

Revisa cosas que no dependen de la dirección de mirada:

| ¿Qué revisa? | ¿Cómo lo detecta? |
|---|---|
| **No hay cara** | Si en el 70% de los últimos 8 frames no se detectó cara |
| **Cara saliéndose** | Si el recuadro de la cara toca los bordes de la imagen |
| **Giro de cabeza** | Si la cara se ve muy alargada o muy aplastada |
| **Muy lejos** | Si la cara ocupa muy poco espacio en la imagen |
| **Ojos no visibles** | Si hay cara pero no se detectan ojos |

Para evitar falsas alarmas:
- Necesita que la violación se mantenga durante varios frames seguidos
- Y que pase un mínimo de tiempo

---

## 6. Las Alertas (`alerts.py`)

Cuando se detecta trampa:
- **Visual**: dibuja un marco rojo grueso alrededor de la pantalla + texto con la razón
- **Sonora**: hace un beep en un hilo aparte para no congelar el video
  - Usa un candado (lock) para que no se encimen varios sonidos

---

## 7. El Flujo Completo

```
1. Se captura un frame de la cámara
2. Se prepara la imagen (resize + grises + CLAHE)
3. Se busca la cara
4. Se buscan los ojos (con filtro anti-cejas)
5. Se busca la pupila (3 métodos en cascada)
6. Se suaviza todo (para que no tiemble)
7. ¿Estamos calibrando?
   ├── Sí → guardar muestras y mostrar cuenta regresiva
   └── No → seguir al paso 8
8. Se analiza: ¿mirada fuera? ¿giro? ¿ausencia? ¿ojos cerrados?
9. Se confirma que la violación sea real (varios frames seguidos)
10. ¿Violación confirmada y pasó el cooldown?
    ├── Sí → strike + alarma + alerta visual
    └── No → seguir
11. ¿5 strikes? → cerrar examen
12. Mostrar todo en pantalla
13. Volver al paso 1
```

---

## Créditos

El método de detección de pupila por contornos está inspirado en:
- **[antoinelame/GazeTracking](https://github.com/antoinelame/GazeTracking)** — Licencia MIT

Solo se usa el algoritmo clásico de pupila (filtros + contornos). No se usan las partes que dependen de dlib o modelos de IA.
