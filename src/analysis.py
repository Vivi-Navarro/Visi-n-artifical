import time
from collections import deque

class Analysis:
    def __init__(self, violation_threshold_seconds=2.5):
        self.threshold = violation_threshold_seconds
        self.last_clean_time = time.time()
        self.is_violating = False

        # Historial de ausencia de rostro para evitar falsos positivos por frames perdidos
        self._face_absence_history = deque(maxlen=6)   # ventana de 6 frames
        self._face_absence_threshold = 0.65            # 65% de frames sin cara = alerta

    def evaluate(self, faces, eyes, pupil_centers, frame_shape):
        violations = []
        h, w = frame_shape[:2]

        # ── 1. Ausencia del rostro (con suavizado para evitar falsos por frames perdidos) ──
        face_present = len(faces) > 0
        self._face_absence_history.append(0 if face_present else 1)

        history = list(self._face_absence_history)
        absence_ratio = sum(history) / len(history) if history else 0.0

        if absence_ratio >= self._face_absence_threshold:
            violations.append("AUSENCIA DE ROSTRO")

        for (x, y, fw, fh) in faces:

            # ── 2. Salida parcial del rostro ──
            margin = int(min(w, h) * 0.04)   # margen relativo al tamaño del frame
            if x < margin or y < margin or (x + fw) > (w - margin) or (y + fh) > (h - margin):
                violations.append("ROSTRO PARCIALMENTE FUERA")

            # ── 3. Giro de cabeza: heurística mejorada ──
            # El aspect ratio SOLO es útil combinado con el tamaño relativo al frame.
            # Un rostro de perfil es mucho más angosto relativo a su altura.
            aspect_ratio = fh / fw if fw > 0 else 1.0
            face_area_ratio = (fw * fh) / (w * h)

            # Si la cara es pequeña en el frame, probablemente está lejos o de lado
            # Aspect ratio < 0.75: cara muy estrecha (perfil)
            # Aspect ratio > 1.7: cara muy alta y estrecha (también perfil o cabeza inclinada)
            if aspect_ratio < 0.75 or aspect_ratio > 1.7:
                violations.append("GIRO DE CABEZA EXCESIVO")

            # Si el área del rostro cayó drásticamente (se alejó o giró), también es sospechoso
            if face_area_ratio < 0.02:
                violations.append("ROSTRO MUY PEQUEÑO / ALEJADO")

            # ── 4. Ausencia de ojos detectados en un rostro presente ──
            # Si hay cara pero no se detectan ojos, suele significar que está mirando abajo
            # (leyendo apuntes) o de lado.
            if len(eyes) == 0 and len(faces) > 0:
                violations.append("OJOS NO DETECTADOS (POSIBLE MIRADA ABAJO)")

        # ── 5. Dirección de la mirada / pupilas desviadas ──
        if len(eyes) > 0 and pupil_centers:
            for i, (px, py) in enumerate(pupil_centers):
                if i < len(eyes):
                    ex, ey, ew, eh = eyes[i]
                    if ew == 0:
                        continue
                    rel_x = px / ew
                    rel_y = py / eh if eh > 0 else 0.5

                    # Horizontal: mirada lateral
                    if rel_x < 0.33 or rel_x > 0.67:
                        violations.append("MIRADA FUERA DE PANTALLA")
                    # Vertical: mirada abajo (leer apuntes)
                    elif rel_y > 0.68:
                        violations.append("MIRADA HACIA ABAJO")

        # ── Evaluación temporal ──
        if not violations:
            self.last_clean_time = time.time()
            self.is_violating = False
            return False, []
        else:
            current_violation_duration = time.time() - self.last_clean_time
            if current_violation_duration > self.threshold:
                self.is_violating = True
                # Deduplicar violaciones
                unique_violations = list(dict.fromkeys(violations))
                return True, unique_violations
            return False, list(dict.fromkeys(violations))