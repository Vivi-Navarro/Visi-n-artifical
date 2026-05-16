import cv2
import numpy as np
import time
from collections import deque

class Detector:
    def __init__(self, face_cascade_path, eye_cascade_path):
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade  = cv2.CascadeClassifier(eye_cascade_path)

        self._gaze_off_since          = None
        self._last_blink_time         = None
        self._blink_tolerance_sec      = 0.25   # parpadeo real ~150ms, margen mínimo
        self._suspicious_threshold_sec = 1.0   # dispara strike tras 1s sostenida (era 2.0)

        # Buffer más corto = reacciona más rápido
        self._gaze_history = deque(maxlen=4)   # era 8

    def detect_face(self, gray_frame):
        # scaleFactor más bajo = más detecciones pero más lento; 1.1 es mejor balance
        return self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)   # ignorar caras muy pequeñas / ruido
        )

    def detect_eyes(self, gray_face_roi):
        eyes = self.eye_cascade.detectMultiScale(
            gray_face_roi,
            scaleFactor=1.05,
            minNeighbors=4,    # era 3, más estricto reduce falsos positivos
            minSize=(20, 20),
            maxSize=(120, 120)
        )
        h = gray_face_roi.shape[0]
        # Solo ojos en mitad superior del rostro
        eyes = [e for e in eyes if e[1] < h // 2]
        # Ordenar de izquierda a derecha para consistencia
        eyes = sorted(eyes, key=lambda e: e[0])
        # Máximo 2 ojos
        return eyes[:2]

    def get_pupil_center(self, eye_gray_roi):
        """
        Método mejorado: combina umbralización adaptativa + blob detection
        para ser robusto a variaciones de iluminación.
        """
        if eye_gray_roi.size == 0:
            return None

        # 1. Normalizar iluminación del ROI del ojo
        eye_eq = cv2.equalizeHist(eye_gray_roi)
        eye_blur = cv2.GaussianBlur(eye_eq, (7, 7), 0)

        h, w = eye_blur.shape
        min_area = (h * w) * 0.03
        max_area = (h * w) * 0.45

        # 2. Umbral adaptativo: mejor que percentil fijo bajo distintas luces
        thresh_adapt = cv2.adaptiveThreshold(
            eye_blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=4
        )

        # 3. Morfología para limpiar ruido pequeño
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh_clean = cv2.morphologyEx(thresh_adapt, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            area = cv2.contourArea(contour)
            if not (min_area < area < max_area):
                continue

            # Verificar circularidad: pupila es aproximadamente circular
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.25:   # descartar formas muy alargadas
                continue

            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Asegurar que el centro no esté en el borde del ROI
            margin = max(3, int(min(h, w) * 0.05))
            if margin < cx < w - margin and margin < cy < h - margin:
                return (cx, cy)

        return None

    def get_gaze_direction(self, pupil_center, eye_roi_shape):
        if pupil_center is None:
            return None
        px, py = pupil_center
        h, w = eye_roi_shape[:2]
        ratio_x = px / w
        ratio_y = py / h

        # Umbral más fino que antes (era 0.30/0.70)
        if ratio_x < 0.33:
            return "left"
        elif ratio_x > 0.67:
            return "right"
        # Detectar mirada arriba/abajo (común al leer apuntes)
        elif ratio_y < 0.30:
            return "up"
        elif ratio_y > 0.70:
            return "down"
        return "center"

    def analyze_gaze(self, pupils_and_rois):
        directions = []
        for p, s in pupils_and_rois:
            d = self.get_gaze_direction(p, s)
            if d is not None:
                directions.append(d)

        if not directions:
            return {'is_suspicious': False, 'direction': 'unknown', 'confidence': 0.0}

        off_center = [d for d in directions if d != 'center']

        # Sin desviación
        if not off_center:
            self._gaze_history.append('center')
            return {'is_suspicious': False, 'direction': 'center', 'confidence': 0.0}

        confidence = len(off_center) / len(directions)

        # Dirección dominante
        from collections import Counter
        dominant = Counter(off_center).most_common(1)[0][0]

        self._gaze_history.append(dominant)

        # Suavizado: confirmar sospecha solo si mayoría del historial concuerda
        history_list = list(self._gaze_history)
        history_off = [d for d in history_list if d != 'center']
        smoothed_confidence = len(history_off) / len(history_list) if history_list else 0.0

        # Umbral bajado a 0.55 (era 0.6) y se usa confianza suavizada
        is_suspicious = smoothed_confidence >= 0.55 and confidence >= 0.5

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
            self._gaze_history.clear()  # reset historial tras strike
            return True, gaze

        return False, gaze