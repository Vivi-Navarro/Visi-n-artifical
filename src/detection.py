import cv2
import numpy as np
import time

class Detector:
    def __init__(self, face_cascade_path, eye_cascade_path):
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

        # --- Control de tiempo para filtrar parpadeos ---
        self._gaze_off_since = None        # momento en que empezó la desviación
        self._blink_tolerance_sec = 0.3    # parpadeo normal dura < 300ms
        self._suspicious_threshold_sec = 1.5  # debe mirar al lado 1.5s seguidos para contar

    # ── detección básica (sin cambios) ────────────────────────────────────────

    def detect_face(self, gray_frame):
        return self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    def detect_eyes(self, gray_face_roi):
        return self.eye_cascade.detectMultiScale(gray_face_roi, 1.1, 10)

    def get_pupil_center(self, eye_gray_roi):
        eye_gray_roi = cv2.GaussianBlur(eye_gray_roi, (7, 7), 0)

        threshold_val = int(np.percentile(eye_gray_roi, 20))
        threshold_val = min(threshold_val, 60)

        _, threshold = cv2.threshold(eye_gray_roi, threshold_val, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        h, w = eye_gray_roi.shape
        min_area = (h * w) * 0.02
        max_area = (h * w) * 0.50

        for contour in contours:
            area = cv2.contourArea(contour)
            if not (min_area < area < max_area):
                continue
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            margin = 3
            if margin < cx < w - margin and margin < cy < h - margin:
                return (cx, cy)
        return None

    # ── dirección de mirada ───────────────────────────────────────────────────

    def get_gaze_direction(self, pupil_center, eye_roi_shape):
        if pupil_center is None:
            return None          # None = no hay datos (parpadeo u oclusión)
        px, _ = pupil_center
        ratio = px / eye_roi_shape[1]
        if ratio < 0.35:
            return "left"
        elif ratio > 0.65:
            return "right"
        return "center"

    def analyze_gaze(self, pupils_and_rois):
        directions = [
            self.get_gaze_direction(p, s)
            for p, s in pupils_and_rois
            if self.get_gaze_direction(p, s) is not None  # ignora frames sin pupila
        ]

        if not directions:
            # Sin datos → probablemente parpadeo, no es sospechoso
            return {'is_suspicious': False, 'direction': 'unknown', 'confidence': 0.0}

        off_center = [d for d in directions if d != 'center']
        confidence = len(off_center) / len(directions)
        dominant = max(set(off_center), key=off_center.count) if off_center else 'center'

        return {
            'is_suspicious': confidence >= 0.6,
            'direction': dominant,
            'confidence': confidence
        }

    # ── filtro anti-parpadeo ──────────────────────────────────────────────────

    def should_trigger_strike(self, pupils_and_rois):
        """
        Retorna True SOLO si la mirada lleva desviada más de
        `_suspicious_threshold_sec` segundos de forma continua.

        Un frame sin pupila (parpadeo) NO reinicia el contador,
        siempre que dure menos de `_blink_tolerance_sec`.

        Returns:
            (bool trigger_strike, dict gaze_info)
        """
        gaze = self.analyze_gaze(pupils_and_rois)
        now = time.time()

        if gaze['direction'] == 'unknown':
            # Sin pupilas detectadas: puede ser parpadeo
            # Toleramos hasta _blink_tolerance_sec sin reiniciar el contador
            if self._gaze_off_since is not None:
                elapsed_no_data = now - self._gaze_off_since
                if elapsed_no_data > self._blink_tolerance_sec + self._suspicious_threshold_sec:
                    # Llevan demasiado tiempo sin datos → resetear para evitar
                    # acumular tiempo de oclusión prolongada (mano frente a cámara, etc.)
                    self._gaze_off_since = None
            # No modificamos _gaze_off_since: el reloj sigue corriendo si ya estaba
            return False, gaze

        if not gaze['is_suspicious']:
            # Mirando al frente → resetear contador
            self._gaze_off_since = None
            return False, gaze

        # Mirada desviada confirmada
        if self._gaze_off_since is None:
            self._gaze_off_since = now   # empezar a contar

        elapsed = now - self._gaze_off_since

        if elapsed >= self._suspicious_threshold_sec:
            self._gaze_off_since = None  # resetear para no contar el mismo evento dos veces
            return True, gaze

        return False, gaze