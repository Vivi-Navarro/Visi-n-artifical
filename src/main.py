import cv2
import os
from capture import Camera
from detection import Detector
from analysis import Analysis
from alerts import Alerter

LIMITE_OPORTUNIDADES = 5
DEBUG_MODE = True   # Cambia a False para producción (oculta overlay técnico)

def draw_strikes(frame, count, limit=LIMITE_OPORTUNIDADES):
    h, w = frame.shape[:2]
    for i in range(limit):
        color = (0, 0, 255) if i < count else (200, 200, 200)
        cx = w - 30 - (limit - 1 - i) * 28
        cv2.circle(frame, (cx, 30), 10, color, -1)
    label = f"Strikes: {count}/{limit}"
    cv2.putText(frame, label, (w - 200, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def draw_debug_overlay(frame, gaze, violations, elapsed_violation):
    """Muestra info técnica útil durante desarrollo."""
    if not DEBUG_MODE:
        return
    h, w = frame.shape[:2]
    info_lines = [
        f"Gaze: {gaze.get('direction','?')} ({gaze.get('confidence',0):.2f})",
        f"Suspicious: {gaze.get('is_suspicious', False)}",
        f"Violation time: {elapsed_violation:.1f}s",
        f"Violations: {', '.join(violations) if violations else 'None'}",
    ]
    y = h - (len(info_lines) * 22) - 10
    # Fondo semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, y - 5), (450, h - 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    for line in info_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1)
        y += 22

def show_exam_closed(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "EXAMEN CERRADO",
                (frame.shape[1]//2 - 200, frame.shape[0]//2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    cv2.putText(frame, "Se detectaron demasiadas violaciones",
                (frame.shape[1]//2 - 260, frame.shape[0]//2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Monitor de Examen", frame)
    cv2.waitKey(3000)

def main():
    face_path = os.path.join("data", "cascades", "haarcascade_frontalface_default.xml")
    eye_path  = os.path.join("data", "cascades", "haarcascade_eye.xml")

    if not os.path.exists(face_path) or not os.path.exists(eye_path):
        print("Error: No se encontraron los archivos Haar Cascade en data/cascades/")
        print("Descarga: haarcascade_frontalface_default.xml y haarcascade_eye.xml")
        return

    cam = Camera()
    det = Detector(face_path, eye_path)
    ana = Analysis(violation_threshold_seconds=2.5)
    alt = Alerter()

    strikes = 0
    exam_closed = False

    print("Sistema iniciado. Presiona 'q' para salir, 'd' para toggle debug.")
    global DEBUG_MODE

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        # Preprocesamiento mejorado: CLAHE en lugar de equalizeHist global
        # CLAHE preserva mejor el contraste local (iluminación variable, gafas, etc.)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        faces = det.detect_face(gray)
        pupils_and_rois = []
        all_eyes = []

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 180, 0), 2)
            roi_gray  = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = det.detect_eyes(roi_gray)
            all_eyes.extend(eyes)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
                eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                pupil = det.get_pupil_center(eye_roi)
                pupils_and_rois.append((pupil, eye_roi.shape))

                if pupil:
                    px, py = pupil
                    cv2.circle(roi_color, (ex + px, ey + py), 3, (0, 0, 255), -1)

        # ── Canal 1: Detección de gaze (dentro de Detector) ──
        gaze_trigger, gaze = det.should_trigger_strike(pupils_and_rois)

        # ── Canal 2: Análisis de postura / presencia (Analysis) ──
        # Extraer centros de pupila para Analysis
        pupil_centers_only = [p for (p, _) in pupils_and_rois if p is not None]
        analysis_trigger, violations = ana.evaluate(faces, all_eyes, pupil_centers_only, frame.shape)

        # Tiempo en violación (para debug overlay)
        import time
        elapsed_violation = time.time() - ana.last_clean_time if ana.is_violating else 0.0

        draw_debug_overlay(frame, gaze, violations, elapsed_violation)

        # ── Strike si CUALQUIERA de los dos canales detecta trampa ──
        triggered = gaze_trigger or analysis_trigger

        if triggered:
            strikes += 1
            reason = gaze['direction'] if gaze_trigger else violations[0] if violations else "análisis"
            print(f"[STRIKE {strikes}/{LIMITE_OPORTUNIDADES}] Razón: {reason}")
            alt.trigger_sound_alert()

            # Mostrar alerta visual breve en frame
            if violations:
                alt.trigger_visual_alert(frame, violations[:2])
            elif gaze_trigger:
                alt.trigger_visual_alert(frame, [f"GAZE: {gaze['direction'].upper()}"])

            if strikes >= LIMITE_OPORTUNIDADES:
                exam_closed = True
                show_exam_closed(frame)
                break

        draw_strikes(frame, strikes)
        cv2.imshow("Monitor de Examen", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            DEBUG_MODE = not DEBUG_MODE
            print(f"[DEBUG] {'ON' if DEBUG_MODE else 'OFF'}")

    cam.release()
    cv2.destroyAllWindows()

    if exam_closed:
        print(f"\n[SISTEMA] Examen cerrado automáticamente por {LIMITE_OPORTUNIDADES} strikes.")

if __name__ == "__main__":
    main()