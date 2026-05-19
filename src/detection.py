# Este archivo se encarga de detectar caras, ojos y pupilas
# Es el cerebro de la visión del sistema

import cv2
import numpy as np
import time
from collections import deque, Counter

class Detector:
    # Clase principal que detecta todo lo visual

    def __init__(self, face_cascade_path, eye_cascade_path):
        # Cargamos los archivos XML que OpenCV usa para detectar caras y ojos
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade  = cv2.CascadeClassifier(eye_cascade_path)

        # Control de tiempo para saber cuánto tiempo llevas mirando fuera
        self._gaze_off_since           = None
        self._last_blink_time          = None
        # Tolerancia para parpadeos (no contar como trampa si parpadeas)
        self._blink_tolerance_sec      = 1.0
        # Cuántos segundos mirando fuera para que cuente como sospechoso
        self._suspicious_threshold_sec = 0.6
        # Umbral más corto cuando la mirada es extrema (ojos muy al lado)
        self._extreme_threshold_sec    = 0.3

        # Historial de las últimas 6 direcciones de mirada
        self._gaze_history = deque(maxlen=6)

        # Suavizado de la posición de los ojos (para que no brinque el recuadro)
        self._smoothed_eyes = [None, None]
        self._ema_alpha = 0.4

        # Suavizado de la posición de la pupila (más fuerte porque es más ruidosa)
        self._smoothed_pupils    = [None, None]
        self._last_pupil_time    = [0.0, 0.0]
        # 30% nuevo + 70% anterior = movimiento suave
        self._pupil_ema_alpha    = 0.30
        # Si la pupila se mueve más del 40% del tamaño del ojo, es movimiento real
        self._pupil_jump_ratio   = 0.40
        # Mantener la última posición de pupila por 0.2 segundos si falla
        self._pupil_persist_sec  = 0.20

        # Datos de calibración (la posición "normal" de las pupilas)
        self._baseline_pupil = []
        self._calibrated = False

        # Qué tan lejos puede estar la pupila del centro antes de ser sospechoso
        self._gaze_tolerance = 0.13
        # Qué tan lejos es "exagerado" (ojos MUY a un lado sin mover la cara)
        self._extreme_gaze_tolerance = 0.28

    # ============================
    # DETECCIÓN DE CARA
    # ============================
    def detect_face(self, gray_frame):
        # Busca caras en la imagen usando Haar Cascade
        return self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

    # ============================
    # DETECCIÓN DE OJOS
    # ============================
    def detect_eyes(self, gray_face_roi):
        # Busca ojos dentro de la zona de la cara
        eyes = self.eye_cascade.detectMultiScale(
            gray_face_roi,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(20, 20),
            maxSize=(120, 120)
        )
        h = gray_face_roi.shape[0]
        # Solo aceptamos ojos que estén entre el 18% y 55% de la altura de la cara
        # (para no confundir cejas o nariz con ojos)
        eyes = [e for e in eyes if (h * 0.18) < e[1] < (h * 0.55)]
        # Verificamos que de verdad parezca un ojo (tiene parte oscura en el centro)
        eyes = [e for e in eyes if self._looks_like_eye(gray_face_roi, e)]
        # Ordenamos de izquierda a derecha y nos quedamos con máximo 2
        eyes = sorted(eyes, key=lambda e: e[0])
        eyes = eyes[:2]
        # Aplicamos suavizado para que no brinquen
        return self._smooth_eyes(eyes)

    def _looks_like_eye(self, gray_face_roi, eye_box):
        # Verifica si una zona realmente parece un ojo
        # Un ojo real tiene el centro más oscuro (la pupila) que los bordes
        # Una ceja es oscura de forma uniforme, así que no pasa este filtro
        ex, ey, ew, eh = eye_box
        if ew < 10 or eh < 10:
            return False
        roi = gray_face_roi[ey:ey+eh, ex:ex+ew]
        if roi.size == 0:
            return False
        h, w = roi.shape
        # Tomamos el centro de la zona
        cy0, cy1 = h // 3, 2 * h // 3
        cx0, cx1 = w // 4, 3 * w // 4
        center = roi[cy0:cy1, cx0:cx1]
        # Tomamos los bordes de arriba y abajo
        edges = np.concatenate([roi[:h//4, :].flatten(), roi[3*h//4:, :].flatten()])
        if center.size == 0 or edges.size == 0:
            return False
        # El punto más oscuro del centro debe ser al menos 12 tonos más oscuro que los bordes
        center_min = float(np.min(center))
        edges_mean = float(np.mean(edges))
        return center_min < edges_mean - 12

    def _smooth_eyes(self, new_eyes):
        # Suaviza la posición de los ojos para que no tiemblen
        # Si el ojo se movió poquito -> promediamos con la posición anterior
        # Si se movió mucho -> aceptamos (es un movimiento real, como girar la cabeza)
        smoothed = []
        for i in range(min(len(new_eyes), 2)):
            new = tuple(int(v) for v in new_eyes[i])
            prev = self._smoothed_eyes[i]
            if prev is None:
                self._smoothed_eyes[i] = new
                smoothed.append(new)
                continue
            # Calculamos qué tanto se movió
            dist = abs(new[0] - prev[0]) + abs(new[1] - prev[1])
            jump_threshold = (prev[2] + prev[3]) // 2
            if dist > jump_threshold:
                # Se movió mucho: aceptar tal cual
                self._smoothed_eyes[i] = new
                smoothed.append(new)
            else:
                # Se movió poquito: promediar para suavizar
                a = self._ema_alpha
                sm = tuple(int(a * new[k] + (1 - a) * prev[k]) for k in range(4))
                self._smoothed_eyes[i] = sm
                smoothed.append(sm)
        return smoothed


    # DETECCIÓN DE PUPILA

    # Usa 3 métodos, uno tras otro, hasta que alguno funcione:
    # 1. HoughCircles (busca círculos)
    # 2. Pipeline estilo GazeTracking (filtros + contornos)
    # 3. Contornos adaptativos (último recurso)

    def get_pupil_center(self, eye_gray_roi, eye_index=None):
        # Busca el centro de la pupila dentro de la zona del ojo
        if eye_gray_roi.size == 0:
            return None
        h, w = eye_gray_roi.shape
        if h < 12 or w < 12:
            return None

        # Checamos si el ojo está cerrado (la imagen se ve muy uniforme verticalmente)
        col_means = np.mean(eye_gray_roi, axis=1)
        vertical_std = float(np.std(col_means))
        if vertical_std < 8.0:
            # Probablemente el ojo está cerrado
            return None

        # Si ya teníamos una pupila anterior, la usamos como referencia
        prev_pupil = None
        if eye_index is not None and 0 <= eye_index < 2:
            prev_pupil = self._smoothed_pupils[eye_index]

        # --- Método 1: buscar círculos con HoughCircles ---
        eye_eq   = cv2.equalizeHist(eye_gray_roi)
        eye_blur = cv2.GaussianBlur(eye_eq, (7, 7), 1.5)

        # Tamaños mínimo y máximo del círculo a buscar
        min_r = max(2, int(min(h, w) * 0.10))
        max_r = max(min_r + 2, int(min(h, w) * 0.35))

        circles = cv2.HoughCircles(
            eye_blur, cv2.HOUGH_GRADIENT,
            dp=1.0, minDist=max(8, w // 2),
            param1=80, param2=15,
            minRadius=min_r, maxRadius=max_r
        )

        # Zona válida para la pupila (no muy arriba ni muy abajo = párpados)
        cy_min_allowed = int(h * 0.20)
        cy_max_allowed = int(h * 0.80)

        # Revisamos cada círculo encontrado
        candidates = []
        if circles is not None and len(circles) > 0:
            margin = max(2, min_r)
            for c in circles[0]:
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                # Verificamos que esté dentro de los límites
                if not (margin <= cx <= w - margin):
                    continue
                if not (cy_min_allowed <= cy <= cy_max_allowed):
                    continue
                # Verificamos que sea un punto oscuro (la pupila es oscura)
                y0, y1 = max(0, cy - 2), min(h, cy + 3)
                x0, x1 = max(0, cx - 2), min(w, cx + 3)
                patch = eye_gray_roi[y0:y1, x0:x1]
                if patch.size == 0:
                    continue
                mean = float(np.mean(patch))
                # Si es muy claro, no es pupila
                if mean > 90:
                    continue
                candidates.append((cx, cy, mean))

        if candidates:
            if prev_pupil is not None:
                # Elegimos el candidato más cercano al anterior (continuidad)
                px, py = prev_pupil
                best = min(candidates,
                           key=lambda c: (c[0] - px) ** 2 + (c[1] - py) ** 2)
            else:
                # Primera vez: elegimos el más oscuro
                best = min(candidates, key=lambda c: c[2])
            return (best[0], best[1])

        # --- Método 2: pipeline estilo GazeTracking ---
        gt_pupil = self._pupil_gazetracking_style(
            eye_gray_roi, cy_min_allowed, cy_max_allowed, prev_pupil)
        if gt_pupil is not None:
            return gt_pupil

        # --- Método 3: contornos adaptativos (último recurso) ---
        return self._pupil_by_contour(eye_blur, h, w, eye_gray_roi,
                                       cy_min_allowed, cy_max_allowed, prev_pupil)

    def _pupil_gazetracking_style(self, eye_gray_roi, cy_min, cy_max, prev_pupil):
        # Método inspirado en el proyecto GazeTracking (MIT License)
        # Usa filtros de imagen + erosión + threshold para encontrar la pupila
        h, w = eye_gray_roi.shape
        # Filtro bilateral: suaviza pero conserva los bordes del iris
        filtered = cv2.bilateralFilter(eye_gray_roi, 10, 15, 15)

        # Erosión 3 veces: elimina pestañas y ruido fino
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(filtered, kernel, iterations=3)

        margin = max(3, int(min(h, w) * 0.05))
        min_area = (h * w) * 0.02
        max_area = (h * w) * 0.55

        # Probamos 3 niveles de umbral y juntamos todos los candidatos
        all_candidates = []
        for thresh_val in (40, 50, 60):
            _, binary = cv2.threshold(eroded, thresh_val, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            # Nos quedamos con los 3 contornos más grandes
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
            for contour in contours:
                area = cv2.contourArea(contour)
                if not (min_area < area < max_area):
                    continue
                # Verificamos que sea redondo (la pupila es circular)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < 0.25:
                    continue
                # Calculamos el centro del contorno
                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # Verificamos que esté en zona válida
                if not (margin < cx < w - margin):
                    continue
                if not (cy_min <= cy <= cy_max):
                    continue
                # Verificamos que sea oscuro
                y0, y1 = max(0, cy - 2), min(h, cy + 3)
                x0, x1 = max(0, cx - 2), min(w, cx + 3)
                patch = eye_gray_roi[y0:y1, x0:x1]
                if patch.size == 0:
                    continue
                mean = float(np.mean(patch))
                if mean > 90:
                    continue
                all_candidates.append((cx, cy, mean))

        if not all_candidates:
            return None

        if prev_pupil is not None:
            # Elegimos el más cercano al anterior
            px, py = prev_pupil
            best = min(all_candidates,
                       key=lambda c: (c[0] - px) ** 2 + (c[1] - py) ** 2)
        else:
            # Sin referencia: el más oscuro
            best = min(all_candidates, key=lambda c: c[2])
        return (best[0], best[1])

    def _pupil_by_contour(self, eye_blur, h, w, eye_gray_roi=None,
                          cy_min_allowed=None, cy_max_allowed=None, prev_pupil=None):
        # Último recurso: usa threshold adaptativo + contornos
        min_area = (h * w) * 0.03
        max_area = (h * w) * 0.45
        if cy_min_allowed is None:
            cy_min_allowed = int(h * 0.20)
        if cy_max_allowed is None:
            cy_max_allowed = int(h * 0.80)

        # Threshold adaptativo para separar la pupila del fondo
        thresh_adapt = cv2.adaptiveThreshold(
            eye_blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=4
        )
        # Limpiamos ruido con morfología
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh_clean = cv2.morphologyEx(thresh_adapt, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        candidates = []
        margin = max(3, int(min(h, w) * 0.05))
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (min_area < area < max_area):
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            # Verificamos circularidad
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.25:
                continue
            # Centro del contorno
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if not (margin < cx < w - margin):
                continue
            if not (cy_min_allowed <= cy <= cy_max_allowed):
                continue
            # Verificamos oscuridad
            if eye_gray_roi is not None:
                y0, y1 = max(0, cy - 2), min(h, cy + 3)
                x0, x1 = max(0, cx - 2), min(w, cx + 3)
                patch = eye_gray_roi[y0:y1, x0:x1]
                if patch.size == 0:
                    continue
                mean = float(np.mean(patch))
                if mean > 90:
                    continue
            else:
                mean = 0.0
            candidates.append((cx, cy, mean))

        if not candidates:
            return None
        if prev_pupil is not None:
            px, py = prev_pupil
            best = min(candidates, key=lambda c: (c[0] - px) ** 2 + (c[1] - py) ** 2)
        else:
            best = min(candidates, key=lambda c: c[2])
        return (best[0], best[1])

    # SUAVIZADO DE PUPILA

    def smooth_pupil(self, pupil, eye_index, eye_roi_shape):
        # Suaviza la posición de la pupila para que no tiemble
        # Si se movió poquito -> promediamos (anti-temblor)
        # Si se movió mucho -> aceptamos (es movimiento real)
        # Si no la detectamos pero había una reciente -> la mantenemos un momento
        if eye_index >= 2 or eye_index < 0:
            return pupil
        now = time.time()
        h, w = eye_roi_shape[:2]
        prev = self._smoothed_pupils[eye_index]

        if pupil is None:
            # No se detectó pupila: mantener la anterior si es reciente
            if prev is not None and (now - self._last_pupil_time[eye_index]) < self._pupil_persist_sec:
                return prev
            self._smoothed_pupils[eye_index] = None
            return None

        self._last_pupil_time[eye_index] = now

        if prev is None:
            # Primera detección de pupila
            self._smoothed_pupils[eye_index] = pupil
            return pupil

        # Calculamos qué tanto se movió respecto al tamaño del ojo
        dx = pupil[0] - prev[0]
        dy = pupil[1] - prev[1]
        dist_norm = (abs(dx) / max(1, w)) + (abs(dy) / max(1, h))

        if dist_norm > self._pupil_jump_ratio:
            # Se movió mucho: aceptar como movimiento real
            self._smoothed_pupils[eye_index] = pupil
            return pupil

        # Se movió poquito: promediar para suavizar
        a = self._pupil_ema_alpha
        sm = (int(a * pupil[0] + (1 - a) * prev[0]),
              int(a * pupil[1] + (1 - a) * prev[1]))
        self._smoothed_pupils[eye_index] = sm
        return sm

    def reset_pupil_smoothing(self):
        # Limpia el suavizado de pupilas (se usa al recalibrar)
        self._smoothed_pupils = [None, None]
        self._last_pupil_time = [0.0, 0.0]

    # ============================
    # CALIBRACIÓN
    # ============================
    # Al inicio, el usuario mira al centro 5 segundos
    # El sistema aprende dónde están sus pupilas "normalmente"
    # Eso se llama el "baseline"

    def reset_calibration(self):
        # Reinicia la calibración (se usa con la tecla 'r')
        self._calibrated = False
        self._baseline_pupil = []
        self.reset_pupil_smoothing()
        self._smoothed_eyes = [None, None]
        self._gaze_history.clear()
        if hasattr(self, '_cal_buffer'):
            del self._cal_buffer

    def add_calibration_sample(self, pupils_and_rois):
        # Guarda una muestra de calibración
        # Solo la guardamos si detectamos AMBOS ojos (para que sea preciso)
        if not hasattr(self, '_cal_buffer'):
            self._cal_buffer = [[], []]
        if len(pupils_and_rois) < 2:
            return
        valid = all(p is not None for (p, _) in pupils_and_rois[:2])
        if not valid:
            return
        for i, (p, s) in enumerate(pupils_and_rois[:2]):
            eh, ew = s[0], s[1]
            if ew == 0 or eh == 0:
                continue
            # Guardamos la posición como proporción (0 a 1) del tamaño del ojo
            self._cal_buffer[i].append((p[0] / ew, p[1] / eh))

    def finish_calibration(self):
        # Termina la calibración calculando el promedio (mediana) de las muestras
        if not hasattr(self, '_cal_buffer'):
            return False
        baseline = []
        success = True
        for samples in self._cal_buffer:
            if len(samples) < 8:
                # Muy pocas muestras: usamos el centro como fallback
                baseline.append((0.5, 0.5))
                success = False
            else:
                xs = [s[0] for s in samples]
                ys = [s[1] for s in samples]
                baseline.append((float(np.median(xs)), float(np.median(ys))))
        self._baseline_pupil = baseline
        self._calibrated = success
        del self._cal_buffer
        return success

    def calibration_samples_count(self):
        # Cuántas muestras válidas llevamos (el mínimo entre ambos ojos)
        if not hasattr(self, '_cal_buffer'):
            return 0
        return min(len(self._cal_buffer[0]), len(self._cal_buffer[1]))

    def is_calibrated(self):
        # ¿Ya se calibró correctamente?
        return self._calibrated

    def get_baseline(self):
        # Devuelve el baseline (posición "normal" de las pupilas)
        return self._baseline_pupil

    # ============================
    # DIRECCIÓN DE MIRADA
    # ============================
    def get_gaze_direction(self, pupil_center, eye_roi_shape, eye_index=0):
        # Calcula hacia dónde está mirando el usuario
        # Compara la posición actual de la pupila con el baseline
        # Devuelve (dirección, desviación_máxima) para saber qué tan lejos mira
        if pupil_center is None:
            return None, 0.0
        px, py = pupil_center
        h, w = eye_roi_shape[:2]
        if w == 0 or h == 0:
            return None, 0.0
        # Calculamos la posición como proporción (0 a 1)
        ratio_x = px / w
        ratio_y = py / h

        if self._calibrated and eye_index < len(self._baseline_pupil):
            # Con calibración: comparamos con el baseline personal
            base_x, base_y = self._baseline_pupil[eye_index]
            dx = ratio_x - base_x
            dy = ratio_y - base_y
            # La desviación máxima (qué tan lejos está la pupila del centro)
            deviation = max(abs(dx), abs(dy))
            tol = self._gaze_tolerance
            if dx < -tol:
                return "left", deviation
            elif dx > tol:
                return "right", deviation
            elif dy < -tol:
                return "up", deviation
            elif dy > tol:
                return "down", deviation
            return "center", deviation

        # Sin calibración: usamos rangos fijos genéricos
        if ratio_x < 0.33:
            return "left", abs(ratio_x - 0.5)
        elif ratio_x > 0.67:
            return "right", abs(ratio_x - 0.5)
        elif ratio_y < 0.30:
            return "up", abs(ratio_y - 0.5)
        elif ratio_y > 0.70:
            return "down", abs(ratio_y - 0.5)
        return "center", 0.0

    def analyze_gaze(self, pupils_and_rois):
        # Analiza la mirada de ambos ojos y da un resultado combinado
        directions = []
        deviations = []
        for i, (p, s) in enumerate(pupils_and_rois):
            d, dev = self.get_gaze_direction(p, s, eye_index=i)
            if d is not None:
                directions.append(d)
                deviations.append(dev)

        if not directions:
            return {'is_suspicious': False, 'direction': 'unknown',
                    'confidence': 0.0, 'is_extreme': False}

        # La desviación máxima entre todos los ojos
        max_deviation = max(deviations) if deviations else 0.0

        # ¿Algún ojo está mirando fuera del centro?
        off_center = [d for d in directions if d != 'center']
        if not off_center:
            self._gaze_history.append('center')
            return {'is_suspicious': False, 'direction': 'center',
                    'confidence': 0.0, 'is_extreme': False}

        # Calculamos la confianza (qué tan seguro estamos)
        confidence = len(off_center) / len(directions)
        # La dirección más común entre los ojos
        dominant = Counter(off_center).most_common(1)[0][0]
        self._gaze_history.append(dominant)

        # ¿La pupila se movió de forma exagerada? (ojos MUY a un lado)
        is_extreme = max_deviation >= self._extreme_gaze_tolerance

        # Revisamos el historial reciente para suavizar la decisión
        history_list = list(self._gaze_history)
        history_off = [d for d in history_list if d != 'center']
        smoothed_confidence = len(history_off) / len(history_list) if history_list else 0.0

        # Es sospechoso si:
        # - La mirada es extrema (pupila muy lejos del centro)
        # - El historial reciente muestra mucha desviación (>= 40%)
        # - O ambos ojos coinciden en que estás mirando fuera (>= 70%)
        is_suspicious = is_extreme or smoothed_confidence >= 0.4 or confidence >= 0.7
        return {
            'is_suspicious': is_suspicious,
            'direction': dominant,
            'confidence': smoothed_confidence,
            'is_extreme': is_extreme
        }

    def should_trigger_strike(self, pupils_and_rois):
        # Decide si hay que dar un strike por mirada sospechosa
        gaze = self.analyze_gaze(pupils_and_rois)
        now  = time.time()

        if gaze['direction'] == 'unknown':
            # No se pudo analizar la mirada
            self._last_blink_time = now
            return False, gaze

        if self._last_blink_time is not None:
            # Tolerancia post-parpadeo (no castigar justo después de parpadear)
            if now - self._last_blink_time < self._blink_tolerance_sec:
                return False, gaze

        if not gaze['is_suspicious']:
            # No es sospechoso, todo bien
            self._gaze_off_since = None
            return False, gaze

        # Es sospechoso: empezamos a contar el tiempo
        if self._gaze_off_since is None:
            self._gaze_off_since = now

        elapsed = now - self._gaze_off_since
        # Si la mirada es extrema (ojos MUY al lado), usamos umbral más corto
        threshold = self._extreme_threshold_sec if gaze.get('is_extreme') else self._suspicious_threshold_sec
        if elapsed >= threshold:
            # Pasó suficiente tiempo mirando fuera: ¡strike!
            self._gaze_off_since = None
            self._gaze_history.clear()
            return True, gaze
        return False, gaze
