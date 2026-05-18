import time
from collections import deque

class Analysis:
    def __init__(self, violation_threshold_seconds=1.0, consecutive_frames=4):
        self.threshold = violation_threshold_seconds
        self.last_clean_time = time.time()
        self.is_violating = False

        # Suavizado de ausencia de rostro
        self._face_absence_history = deque(maxlen=8)
        self._face_absence_threshold = 0.70  # 70% de frames sin rostro = ausencia

        # Confirmación temporal: cuántos frames consecutivos con violación se requieren
        # antes de aceptarla como real (anti-glitch de Haar Cascade)
        self._consecutive_required = consecutive_frames
        self._consecutive_violating = 0

    def evaluate(self, faces, eyes, pupil_centers, frame_shape):
        """Detecta infracciones que NO dependen de dirección de mirada
        (eso lo hace Detector.should_trigger_strike con baseline calibrado).
        Aquí: ausencia, salida parcial, giro de cabeza, ojos no detectados."""
        violations = []
        h, w = frame_shape[:2]

        # ── 1. Ausencia del rostro (con suavizado) ──
        face_present = len(faces) > 0
        self._face_absence_history.append(0 if face_present else 1)
        history = list(self._face_absence_history)
        absence_ratio = sum(history) / len(history) if history else 0.0
        if absence_ratio >= self._face_absence_threshold:
            violations.append("AUSENCIA DE ROSTRO")

        for (x, y, fw, fh) in faces:
            # ── 2. Salida parcial del rostro ──
            margin = int(min(w, h) * 0.04)
            if x < margin or y < margin or (x + fw) > (w - margin) or (y + fh) > (h - margin):
                violations.append("ROSTRO PARCIALMENTE FUERA")

            # ── 3. Giro de cabeza excesivo ──
            aspect_ratio    = fh / fw if fw > 0 else 1.0
            face_area_ratio = (fw * fh) / (w * h)
            if aspect_ratio < 0.75 or aspect_ratio > 1.7:
                violations.append("GIRO DE CABEZA EXCESIVO")
            if face_area_ratio < 0.02:
                violations.append("ROSTRO MUY PEQUEÑO / ALEJADO")

            # ── 4. Ojos no detectados con rostro presente ──
            # (mirada hacia abajo / cabeza inclinada / ojos cerrados sostenidos)
            if len(eyes) == 0:
                violations.append("OJOS NO DETECTADOS")

        # ── Evaluación temporal con confirmación de frames consecutivos ──
        if not violations:
            self.last_clean_time = time.time()
            self.is_violating = False
            self._consecutive_violating = 0
            return False, []

        # Hay violaciones: incrementar contador de frames consecutivos
        self._consecutive_violating += 1
        unique_violations = list(dict.fromkeys(violations))

        # Aceptar como violación real solo si pasó el threshold de tiempo
        # Y se sostuvo N frames consecutivos
        current_duration = time.time() - self.last_clean_time
        if (current_duration > self.threshold and
            self._consecutive_violating >= self._consecutive_required):
            self.is_violating = True
            return True, unique_violations

        return False, unique_violations
