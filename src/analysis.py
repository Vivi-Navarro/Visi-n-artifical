import time

class Analysis:
    def __init__(self, violation_threshold_seconds=3):
        self.threshold = violation_threshold_seconds
        self.last_clean_time = time.time()
        self.is_violating = False

    def evaluate(self, faces, eyes, pupil_centers, frame_shape):
        violations = []
        h, w = frame_shape[:2]

        # 1. Ausencia del rostro
        if len(faces) == 0:
            violations.append("AUSENCIA DE ROSTRO")
        
        for (x, y, fw, fh) in faces:
            # 2. Salida parcial del rostro
            margin = 20
            if x < margin or y < margin or (x + fw) > (w - margin) or (y + fh) > (h - margin):
                violations.append("ROSTRO PARCIALMENTE FUERA")

            # 3. Giro excesivo de cabeza (Heurística simple: relación de aspecto)
            # Un rostro frontal suele tener una relación de aspecto cercana a 1.0 - 1.2
            aspect_ratio = fh / fw
            if aspect_ratio < 0.8 or aspect_ratio > 1.6:
                violations.append("GIRO DE CABEZA EXCESIVO")

        # 4. Dirección de la mirada / Ojos fuera de pantalla
        if len(eyes) > 0 and pupil_centers:
            # Si se detectan ojos pero no se encuentran pupilas centrales, puede ser mirada desviada
            # O si el centro de la pupila está muy desplazado en el ROI del ojo.
            for i, (px, py) in enumerate(pupil_centers):
                if i < len(eyes):
                    ex, ey, ew, eh = eyes[i]
                    # Relativo al centro del ojo
                    rel_x = px / ew
                    if rel_x < 0.35 or rel_x > 0.65:
                        violations.append("MIRADA FUERA DE PANTALLA")

        if not violations:
            self.last_clean_time = time.time()
            self.is_violating = False
            return False, []
        else:
            current_violation_duration = time.time() - self.last_clean_time
            if current_violation_duration > self.threshold:
                self.is_violating = True
                return True, violations
            return False, violations
