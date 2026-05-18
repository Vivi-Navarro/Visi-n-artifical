
import cv2
import numpy as np
import time
from collections import deque, Counter

class Detector:
    def __init__(self, face_cascade_path, eye_cascade_path):
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade  = cv2.CascadeClassifier(eye_cascade_path)

        self._gaze_off_since           = None
        self._last_blink_time          = None
        self._blink_tolerance_sec      = 7.9
        self._suspicious_threshold_sec = 0.6   # antes 1.0 — más reactivo

        self._gaze_history = deque(maxlen=6)   # historial un poco más largo

        # Suavizado temporal (EMA) de bounding boxes de ojos
        self._smoothed_eyes = [None, None]
        self._ema_alpha = 0.4

        # Suavizado de pupila (más fuerte que el de bbox porque HoughCircles es ruidoso)
        self._smoothed_pupils    = [None, None]   # (cx, cy) en coords del ROI del ojo
        self._last_pupil_time    = [0.0, 0.0]
        self._pupil_ema_alpha    = 0.30           # 30% nuevo, 70% histórico
        self._pupil_jump_ratio   = 0.40           # >40% del tamaño del ojo = movimiento real
        self._pupil_persist_sec  = 0.20           # mantener última pupila este tiempo si falla detección

        # Calibración
        self._baseline_pupil = []
        self._calibrated = False

        # Tolerancia más estricta — desvía antes
        self._gaze_tolerance = 0.13   # antes 0.18

    def detect_face(self, gray_frame):
        return self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

    def detect_eyes(self, gray_face_roi):
        eyes = self.eye_cascade.detectMultiScale(
            gray_face_roi,
            scaleFactor=1.05,
            minNeighbors=5,        # más estricto: era 4
            minSize=(20, 20),
            maxSize=(120, 120)
        )
        h = gray_face_roi.shape[0]
        # Restringir a banda media (no muy arriba = cejas/frente, no muy abajo = nariz)
        # Ojos suelen estar entre el 20% y el 55% de la altura del rostro.
        eyes = [e for e in eyes if (h * 0.18) < e[1] < (h * 0.55)]
        # Validar que tenga "oscuridad central" (iris/pupila) y no sea ceja/sombra
        eyes = [e for e in eyes if self._looks_like_eye(gray_face_roi, e)]
        eyes = sorted(eyes, key=lambda e: e[0])
        eyes = eyes[:2]
        return self._smooth_eyes(eyes)

    def _looks_like_eye(self, gray_face_roi, eye_box):
        """Heurística: en un ojo real el centro es notablemente más oscuro (pupila/iris)
        que los bordes superior+inferior. Una ceja es una franja uniformemente oscura."""
        ex, ey, ew, eh = eye_box
        if ew < 10 or eh < 10:
            return False
        roi = gray_face_roi[ey:ey+eh, ex:ex+ew]
        if roi.size == 0:
            return False
        h, w = roi.shape
        # Centro del ROI
        cy0, cy1 = h // 3, 2 * h // 3
        cx0, cx1 = w // 4, 3 * w // 4
        center = roi[cy0:cy1, cx0:cx1]
        # Bordes verticales (encima y debajo del centro)
        edges = np.concatenate([roi[:h//4, :].flatten(), roi[3*h//4:, :].flatten()])
        if center.size == 0 or edges.size == 0:
            return False
        # El píxel más oscuro del centro debe ser >= 12 unidades más oscuro que el promedio de bordes
        center_min = float(np.min(center))
        edges_mean = float(np.mean(edges))
        return center_min < edges_mean - 12

    def _smooth_eyes(self, new_eyes):
        """EMA por ojo. Si el nuevo bbox saltó muy lejos del previo, aceptarlo tal cual
        (cambio real, no jitter). Si está cerca, promediar para estabilizar."""
        smoothed = []
        for i in range(min(len(new_eyes), 2)):
            new = tuple(int(v) for v in new_eyes[i])
            prev = self._smoothed_eyes[i]
            if prev is None:
                self._smoothed_eyes[i] = new
                smoothed.append(new)
                continue
            # Distancia entre centros (Manhattan)
            dist = abs(new[0] - prev[0]) + abs(new[1] - prev[1])
            jump_threshold = (prev[2] + prev[3]) // 2  # ~tamaño promedio del ojo
            if dist > jump_threshold:
                # Salto grande: probablemente cambio real (giró la cabeza)
                self._smoothed_eyes[i] = new
                smoothed.append(new)
            else:
                a = self._ema_alpha
                sm = tuple(int(a * new[k] + (1 - a) * prev[k]) for k in range(4))
                self._smoothed_eyes[i] = sm
                smoothed.append(sm)
        # Si esta vez solo se detectó 1 ojo, no resetear el otro slot — útil si el otro vuelve
        return smoothed

    def get_pupil_center(self, eye_gray_roi, eye_index=None):
        """Método principal: HoughCircles. Fallback: contornos+threshold.
        Usa `eye_index` para conocer la pupila previa y elegir el candidato
        más coherente temporalmente (no solo el más oscuro)."""
        if eye_gray_roi.size == 0:
            return None
        h, w = eye_gray_roi.shape
        if h < 12 or w < 12:
            return None

        # ── Detectar ojo cerrado por varianza vertical baja ──
        # Un ojo abierto tiene cambio fuerte vertical (pupila ↔ esclera).
        # Un ojo cerrado es una banda más uniforme.
        col_means = np.mean(eye_gray_roi, axis=1)
        vertical_std = float(np.std(col_means))
        if vertical_std < 8.0:
            return None   # ojo probablemente cerrado / sin iris visible

        # Pupila previa (si existe) para favorecer continuidad temporal
        prev_pupil = None
        if eye_index is not None and 0 <= eye_index < 2:
            prev_pupil = self._smoothed_pupils[eye_index]

        # ── Método 1: HoughCircles ──
        eye_eq   = cv2.equalizeHist(eye_gray_roi)
        eye_blur = cv2.GaussianBlur(eye_eq, (7, 7), 1.5)

        min_r = max(2, int(min(h, w) * 0.10))
        max_r = max(min_r + 2, int(min(h, w) * 0.35))

        circles = cv2.HoughCircles(
            eye_blur, cv2.HOUGH_GRADIENT,
            dp=1.0, minDist=max(8, w // 2),
            param1=80, param2=15,
            minRadius=min_r, maxRadius=max_r
        )

        # Banda vertical donde puede estar la pupila (evita párpados)
        cy_min_allowed = int(h * 0.20)
        cy_max_allowed = int(h * 0.80)

        candidates = []   # cada uno: (cx, cy, darkness)
        if circles is not None and len(circles) > 0:
            margin = max(2, min_r)
            for c in circles[0]:
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                if not (margin <= cx <= w - margin):
                    continue
                if not (cy_min_allowed <= cy <= cy_max_allowed):
                    continue
                # Validar oscuridad sobre el ROI sin equalizar (intensidad real)
                y0, y1 = max(0, cy - 2), min(h, cy + 3)
                x0, x1 = max(0, cx - 2), min(w, cx + 3)
                patch = eye_gray_roi[y0:y1, x0:x1]
                if patch.size == 0:
                    continue
                mean = float(np.mean(patch))
                # La pupila/iris es notablemente oscura — rechazar candidatos claros
                if mean > 90:
                    continue
                candidates.append((cx, cy, mean))

        if candidates:
            if prev_pupil is not None:
                # Continuidad temporal: el más cercano al previo
                px, py = prev_pupil
                best = min(candidates,
                           key=lambda c: (c[0] - px) ** 2 + (c[1] - py) ** 2)
            else:
                # Primer frame / sin previo: el más oscuro
                best = min(candidates, key=lambda c: c[2])
            return (best[0], best[1])

        # ── Método 2: pipeline estilo GazeTracking (clásico, sin ML) ──
        # Más robusto cuando HoughCircles falla (ojos parcialmente cerrados,
        # iluminación lateral, lentes con reflejos).
        gt_pupil = self._pupil_gazetracking_style(
            eye_gray_roi, cy_min_allowed, cy_max_allowed, prev_pupil)
        if gt_pupil is not None:
            return gt_pupil

        # ── Método 3 (fallback): contornos + threshold adaptativo ──
        return self._pupil_by_contour(eye_blur, h, w, eye_gray_roi,
                                       cy_min_allowed, cy_max_allowed, prev_pupil)

    def _pupil_gazetracking_style(self, eye_gray_roi, cy_min, cy_max, prev_pupil):
        """Pipeline adaptado de antoinelame/GazeTracking (MIT, solo OpenCV clásico):
        bilateralFilter → erosión x3 → threshold fijo → contornos → centroide.

        Sin landmarks dlib — opera sobre el ROI ya recortado por Haar Cascade.
        Probamos 3 thresholds y nos quedamos con el candidato más coherente."""
        h, w = eye_gray_roi.shape
        # 1. Bilateral filter preserva los bordes del iris (mejor que GaussianBlur)
        filtered = cv2.bilateralFilter(eye_gray_roi, 10, 15, 15)

        # 2. Erosión múltiple para eliminar pestañas y ruido fino
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(filtered, kernel, iterations=3)

        margin = max(3, int(min(h, w) * 0.05))
        min_area = (h * w) * 0.02
        max_area = (h * w) * 0.55

        # 3. Probar varios thresholds y acumular candidatos
        all_candidates = []   # (cx, cy, darkness)
        for thresh_val in (40, 50, 60):
            _, binary = cv2.threshold(eroded, thresh_val, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            # Ordenar por área descendente; revisamos los top 3
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
            for contour in contours:
                area = cv2.contourArea(contour)
                if not (min_area < area < max_area):
                    continue
                # Validación de circularidad (la pupila es aprox circular)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < 0.25:
                    continue
                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # Posición válida
                if not (margin < cx < w - margin):
                    continue
                if not (cy_min <= cy <= cy_max):
                    continue
                # Oscuridad real (sobre el ROI sin procesar)
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
            px, py = prev_pupil
            best = min(all_candidates,
                       key=lambda c: (c[0] - px) ** 2 + (c[1] - py) ** 2)
        else:
            best = min(all_candidates, key=lambda c: c[2])
        return (best[0], best[1])

    def _pupil_by_contour(self, eye_blur, h, w, eye_gray_roi=None,
                          cy_min_allowed=None, cy_max_allowed=None, prev_pupil=None):
        min_area = (h * w) * 0.03
        max_area = (h * w) * 0.45
        if cy_min_allowed is None:
            cy_min_allowed = int(h * 0.20)
        if cy_max_allowed is None:
            cy_max_allowed = int(h * 0.80)

        thresh_adapt = cv2.adaptiveThreshold(
            eye_blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=4
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh_clean = cv2.morphologyEx(thresh_adapt, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        candidates = []  # (cx, cy, darkness)
        margin = max(3, int(min(h, w) * 0.05))
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (min_area < area < max_area):
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.25:
                continue
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if not (margin < cx < w - margin):
                continue
            if not (cy_min_allowed <= cy <= cy_max_allowed):
                continue
            # Validar oscuridad
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

    # ── Suavizado de pupila ──────────────────────────────────────────────────
    def smooth_pupil(self, pupil, eye_index, eye_roi_shape):
        """Aplica EMA + persistencia temporal al centro de la pupila por ojo.
        - Si el nuevo punto está cerca del anterior → promedio ponderado (anti-jitter).
        - Si saltó mucho → aceptar (movimiento real de mirada).
        - Si no se detectó pupila pero había una reciente → mantenerla brevemente."""
        if eye_index >= 2 or eye_index < 0:
            return pupil
        now = time.time()
        h, w = eye_roi_shape[:2]
        prev = self._smoothed_pupils[eye_index]

        if pupil is None:
            # Pupila no detectada este frame: persistir la anterior si es reciente
            if prev is not None and (now - self._last_pupil_time[eye_index]) < self._pupil_persist_sec:
                return prev
            self._smoothed_pupils[eye_index] = None
            return None

        self._last_pupil_time[eye_index] = now

        if prev is None:
            self._smoothed_pupils[eye_index] = pupil
            return pupil

        # Distancia normalizada al tamaño del ojo
        dx = pupil[0] - prev[0]
        dy = pupil[1] - prev[1]
        dist_norm = (abs(dx) / max(1, w)) + (abs(dy) / max(1, h))

        if dist_norm > self._pupil_jump_ratio:
            # Salto grande: aceptar como movimiento real
            self._smoothed_pupils[eye_index] = pupil
            return pupil

        # Suavizado fuerte
        a = self._pupil_ema_alpha
        sm = (int(a * pupil[0] + (1 - a) * prev[0]),
              int(a * pupil[1] + (1 - a) * prev[1]))
        self._smoothed_pupils[eye_index] = sm
        return sm

    def reset_pupil_smoothing(self):
        """Limpia el estado de suavizado de pupilas (útil al recalibrar)."""
        self._smoothed_pupils = [None, None]
        self._last_pupil_time = [0.0, 0.0]

    # ── Calibración ──────────────────────────────────────────────────────────
    def reset_calibration(self):
        """Permite recalibrar en caliente (tecla 'r')."""
        self._calibrated = False
        self._baseline_pupil = []
        self.reset_pupil_smoothing()
        self._smoothed_eyes = [None, None]
        self._gaze_history.clear()
        if hasattr(self, '_cal_buffer'):
            del self._cal_buffer

    def add_calibration_sample(self, pupils_and_rois):
        """Solo acumula muestras cuando hay AMBAS pupilas detectadas (evita baseline
        sesgado por parpadeo o detección parcial)."""
        if not hasattr(self, '_cal_buffer'):
            self._cal_buffer = [[], []]
        if len(pupils_and_rois) < 2:
            return  # necesita ambos ojos
        valid = all(p is not None for (p, _) in pupils_and_rois[:2])
        if not valid:
            return
        for i, (p, s) in enumerate(pupils_and_rois[:2]):
            eh, ew = s[0], s[1]
            if ew == 0 or eh == 0:
                continue
            self._cal_buffer[i].append((p[0] / ew, p[1] / eh))

    def finish_calibration(self):
        """Promedia el buffer (mediana) y guarda baseline. Si no hubo muestras
        suficientes, deja sin calibrar para que el sistema use umbrales fijos."""
        if not hasattr(self, '_cal_buffer'):
            return False
        baseline = []
        success = True
        for samples in self._cal_buffer:
            if len(samples) < 8:
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
        """Cantidad mínima de muestras acumuladas entre los dos ojos."""
        if not hasattr(self, '_cal_buffer'):
            return 0
        return min(len(self._cal_buffer[0]), len(self._cal_buffer[1]))

    def is_calibrated(self):
        return self._calibrated

    def get_baseline(self):
        return self._baseline_pupil

    # ── Análisis de mirada (usa baseline si está calibrado) ──────────────────
    def get_gaze_direction(self, pupil_center, eye_roi_shape, eye_index=0):
        if pupil_center is None:
            return None
        px, py = pupil_center
        h, w = eye_roi_shape[:2]
        if w == 0 or h == 0:
            return None
        ratio_x = px / w
        ratio_y = py / h

        if self._calibrated and eye_index < len(self._baseline_pupil):
            base_x, base_y = self._baseline_pupil[eye_index]
            dx = ratio_x - base_x
            dy = ratio_y - base_y
            tol = self._gaze_tolerance
            if dx < -tol:
                return "left"
            elif dx > tol:
                return "right"
            elif dy < -tol:
                return "up"
            elif dy > tol:
                return "down"
            return "center"

        # Sin calibración: umbrales fijos clásicos
        if ratio_x < 0.33:
            return "left"
        elif ratio_x > 0.67:
            return "right"
        elif ratio_y < 0.30:
            return "up"
        elif ratio_y > 0.70:
            return "down"
        return "center"

    def analyze_gaze(self, pupils_and_rois):
        directions = []
        for i, (p, s) in enumerate(pupils_and_rois):
            d = self.get_gaze_direction(p, s, eye_index=i)
            if d is not None:
                directions.append(d)

        if not directions:
            return {'is_suspicious': False, 'direction': 'unknown', 'confidence': 0.0}

        off_center = [d for d in directions if d != 'center']
        if not off_center:
            self._gaze_history.append('center')
            return {'is_suspicious': False, 'direction': 'center', 'confidence': 0.0}

        confidence = len(off_center) / len(directions)
        dominant = Counter(off_center).most_common(1)[0][0]
        self._gaze_history.append(dominant)

        history_list = list(self._gaze_history)
        history_off = [d for d in history_list if d != 'center']
        smoothed_confidence = len(history_off) / len(history_list) if history_list else 0.0

        # Más sensible: si UN ojo se desvía consistentemente (smoothed >= 0.4)
        # O si AMBOS ojos coinciden en desviación en este frame (conf >= 0.7)
        is_suspicious = smoothed_confidence >= 0.4 or confidence >= 0.7
        return {
            'is_suspicious': is_suspicious,
            'direction': dominant,
            'confidence': smoothed_confidence
        }

    def should_trigger_strike(self, pupils_and_rois):
        gaze = self.analyze_gaze(pupils_and_rois)
        now  = time.time()

        if gaze['direction'] == 'unknown':
            self._last_blink_time = now
            return False, gaze

        if self._last_blink_time is not None:
            if now - self._last_blink_time < self._blink_tolerance_sec:
                return False, gaze

        if not gaze['is_suspicious']:
            self._gaze_off_since = None
            return False, gaze

        if self._gaze_off_since is None:
            self._gaze_off_since = now

        elapsed = now - self._gaze_off_since
        if elapsed >= self._suspicious_threshold_sec:
            self._gaze_off_since = None
            self._gaze_history.clear()
            return True, gaze
        return False, gaze
