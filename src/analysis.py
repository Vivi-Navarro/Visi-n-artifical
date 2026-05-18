# Este archivo analiza lo que pasa en cada frame y decide si hay trampa

import time
from collections import deque

class Analysis:
    # Clase que revisa si el usuario está haciendo algo sospechoso

    def __init__(self, violation_threshold_seconds=1.0, consecutive_frames=4):
        # Cuántos segundos debe durar una violación para que cuente
        self.threshold = violation_threshold_seconds
        # Momento del último frame limpio (sin problemas)
        self.last_clean_time = time.time()
        # ¿Estamos en medio de una violación?
        self.is_violating = False

        # Historial de los últimos 8 frames para saber si hay cara o no
        self._face_absence_history = deque(maxlen=8)
        # Si el 70% de esos frames no hay cara, se cuenta como ausencia
        self._face_absence_threshold = 0.70

        # Cuántos frames seguidos con violación necesitamos para aceptarla como real
        self._consecutive_required = consecutive_frames
        # Contador de frames seguidos con violación
        self._consecutive_violating = 0

    def evaluate(self, faces, eyes, pupil_centers, frame_shape):
        # Aquí revisamos cosas que NO dependen de la dirección de mirada
        # (eso lo hace detection.py con la calibración)
        violations = []
        h, w = frame_shape[:2]

        # --- 1. ¿Hay cara presente? ---
        face_present = len(faces) > 0
        # Guardamos 0 si hay cara, 1 si no hay
        self._face_absence_history.append(0 if face_present else 1)
        history = list(self._face_absence_history)
        # Calculamos qué porcentaje de frames recientes no tienen cara
        absence_ratio = sum(history) / len(history) if history else 0.0
        if absence_ratio >= self._face_absence_threshold:
            violations.append("AUSENCIA DE ROSTRO")

        for (x, y, fw, fh) in faces:
            # --- 2. ¿La cara se está saliendo de la pantalla? ---
            margin = int(min(w, h) * 0.04)
            if x < margin or y < margin or (x + fw) > (w - margin) or (y + fh) > (h - margin):
                violations.append("ROSTRO PARCIALMENTE FUERA")

            # --- 3. ¿Está girando mucho la cabeza? ---
            # Si la cara se ve muy alargada o muy aplastada, es giro
            aspect_ratio    = fh / fw if fw > 0 else 1.0
            face_area_ratio = (fw * fh) / (w * h)
            if aspect_ratio < 0.75 or aspect_ratio > 1.7:
                violations.append("GIRO DE CABEZA EXCESIVO")
            # Si la cara se ve muy chiquita, está muy lejos
            if face_area_ratio < 0.02:
                violations.append("ROSTRO MUY PEQUEÑO / ALEJADO")

            # --- 4. ¿No se ven los ojos? ---
            if len(eyes) == 0:
                violations.append("OJOS NO DETECTADOS")

        # --- Decidir si es una violación real o solo un glitch ---
        if not violations:
            # Todo bien, reseteamos
            self.last_clean_time = time.time()
            self.is_violating = False
            self._consecutive_violating = 0
            return False, []

        # Hay violaciones: sumamos al contador
        self._consecutive_violating += 1
        # Quitamos duplicados
        unique_violations = list(dict.fromkeys(violations))

        # Solo aceptamos como violación real si pasó suficiente tiempo
        # Y se mantuvo durante suficientes frames seguidos
        current_duration = time.time() - self.last_clean_time
        if (current_duration > self.threshold and
            self._consecutive_violating >= self._consecutive_required):
            self.is_violating = True
            return True, unique_violations

        return False, unique_violations
