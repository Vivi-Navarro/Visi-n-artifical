import cv2
import os
from capture import Camera
from detection import Detector
from analysis import Analysis
from alerts import Alerter

LIMITE_OPORTUNIDADES = 5
DEBUG_MODE = False   # Cambia a False para producción (oculta overlay técnico)

# ── Configuración de ventana ────────────────────────────────────────────────
WINDOW_NAME   = "Monitor"
WINDOW_W      = 320          # ancho de la ventana en pantalla (px)
WINDOW_H      = 240          # alto  de la ventana en pantalla (px)
WINDOW_POS_X  = None         # None = calcular automático (esquina superior derecha)
WINDOW_POS_Y  = 20           # margen desde la parte superior (px)
PROCESS_W     = 640          # resolución interna para procesar (mayor = más preciso, más lento)
PROCESS_H     = 480
# ───────────────────────────────────────────────────────────────────────────

def _init_window():
    """Crea y posiciona la ventana en la esquina superior derecha."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

    # Calcular posición: esquina superior derecha de la pantalla
    try:
        # Obtener resolución de pantalla usando una ventana auxiliar temporal
        tmp = cv2.getWindowImageRect(WINDOW_NAME)
        # fallback si getWindowImageRect no devuelve info de monitor
    except Exception:
        pass

   
    screen_w = 1500   #ajustalo a tu pantalla solo para pruebas
    pos_x = WINDOW_POS_X if WINDOW_POS_X is not None else screen_w - WINDOW_W - 10
    cv2.moveWindow(WINDOW_NAME, pos_x, WINDOW_POS_Y)

def draw_strikes(frame, count, limit=LIMITE_OPORTUNIDADES):
    """HUD minimalista: puntos pequeños en la esquina superior derecha del frame."""
    h, w = frame.shape[:2]
    dot_r      = 5     # radio de cada punto
    dot_gap    = 14    # separación entre puntos
    pad        = 10    # margen desde el borde

    for i in range(limit):
        filled = i < count
        color  = (60, 60, 220) if filled else (70, 70, 70)
        cx = w - pad - dot_r - (limit - 1 - i) * dot_gap
        cy = pad + dot_r
        if filled:
            cv2.circle(frame, (cx, cy), dot_r, color, -1)
        else:
            cv2.circle(frame, (cx, cy), dot_r, color, 1)   # solo contorno si vacío

def draw_debug_overlay(frame, gaze, violations, elapsed_violation):
    """Overlay de debug compacto: una sola línea en la parte inferior."""
    if not DEBUG_MODE:
        return
    h, w = frame.shape[:2]
    direction  = gaze.get('direction', '?')
    confidence = gaze.get('confidence', 0.0)
    viol_short = violations[0][:18] if violations else "ok"
    line = f"{direction} {confidence:.2f} | {viol_short} | {elapsed_violation:.1f}s"

    # Fondo negro semitransparente bajo el texto
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 18), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, line, (4, h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 100), 1)

def show_exam_closed(frame):
    """Pantalla de cierre: escala al tamaño de proceso para legibilidad."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
    cv2.putText(frame, "EXAMEN CERRADO",
                (cx - 195, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 255), 2)
    cv2.putText(frame, "Demasiadas violaciones detectadas",
                (cx - 220, cy + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    display = cv2.resize(frame, (WINDOW_W, WINDOW_H))
    cv2.imshow(WINDOW_NAME, display)
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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))   # crear una sola vez

    _init_window()
    print("Sistema iniciado. Presiona 'q' para salir, 'd' para toggle debug.")
    global DEBUG_MODE

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        # Escalar frame a resolución de proceso (detección más precisa que en miniatura)
        proc = cv2.resize(frame, (PROCESS_W, PROCESS_H))

        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)

        faces = det.detect_face(gray)
        pupils_and_rois = []
        all_eyes = []

        for (x, y, w, h) in faces:
            cv2.rectangle(proc, (x, y), (x + w, y + h), (255, 180, 0), 1)
            roi_gray  = gray[y:y+h, x:x+w]
            roi_color = proc[y:y+h, x:x+w]

            eyes = det.detect_eyes(roi_gray)
            all_eyes.extend(eyes)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 200, 0), 1)
                eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                pupil = det.get_pupil_center(eye_roi)
                pupils_and_rois.append((pupil, eye_roi.shape))

                if pupil:
                    px, py = pupil
                    cv2.circle(roi_color, (ex + px, ey + py), 2, (0, 0, 200), -1)

        gaze_trigger, gaze = det.should_trigger_strike(pupils_and_rois)

        pupil_centers_only = [p for (p, _) in pupils_and_rois if p is not None]
        analysis_trigger, violations = ana.evaluate(faces, all_eyes, pupil_centers_only, proc.shape)

        import time
        elapsed_violation = time.time() - ana.last_clean_time if ana.is_violating else 0.0

        draw_debug_overlay(proc, gaze, violations, elapsed_violation)
        draw_strikes(proc, strikes)

        triggered = gaze_trigger or analysis_trigger

        if triggered:
            strikes += 1
            reason = gaze['direction'] if gaze_trigger else violations[0] if violations else "análisis"
            print(f"[STRIKE {strikes}/{LIMITE_OPORTUNIDADES}] Razón: {reason}")
            alt.trigger_sound_alert()

            if violations:
                alt.trigger_visual_alert(proc, violations[:2])
            elif gaze_trigger:
                alt.trigger_visual_alert(proc, [f"GAZE: {gaze['direction'].upper()}"])

            if strikes >= LIMITE_OPORTUNIDADES:
                exam_closed = True
                show_exam_closed(proc)
                break

        # Mostrar en ventana pequeña (display != resolución de proceso)
        display = cv2.resize(proc, (WINDOW_W, WINDOW_H))
        cv2.imshow(WINDOW_NAME, display)

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