import cv2
import os
import sys
import time
from capture import Camera
from detection import Detector
from analysis import Analysis
from alerts import Alerter

# Fix encoding UTF-8 en consola Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

LIMITE_OPORTUNIDADES = 5
DEBUG_MODE           = True   # arrancar con debug ON para diagnosticar
STRIKE_COOLDOWN_SEC  = 3.0    # tiempo mínimo entre strikes
CALIBRATION_SEC      = 5      # duración fase de calibración inicial

# ── Configuración de ventana ────────────────────────────────────────────────
WINDOW_NAME   = "Monitor"
WINDOW_W      = 800
WINDOW_H      = 600
WINDOW_POS_X  = None
WINDOW_POS_Y  = 20
PROCESS_W     = 640
PROCESS_H     = 480
# ───────────────────────────────────────────────────────────────────────────

def _get_screen_width():
    """Devuelve ancho de pantalla principal. Fallback 1500 si falla."""
    try:
        import ctypes
        return int(ctypes.windll.user32.GetSystemMetrics(0))
    except Exception:
        return 1500

def _init_window():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)
    screen_w = _get_screen_width()
    pos_x = WINDOW_POS_X if WINDOW_POS_X is not None else screen_w - WINDOW_W - 10
    cv2.moveWindow(WINDOW_NAME, pos_x, WINDOW_POS_Y)

def draw_strikes(frame, count, limit=LIMITE_OPORTUNIDADES):
    h, w = frame.shape[:2]
    dot_r, dot_gap, pad = 8, 22, 14
    for i in range(limit):
        filled = i < count
        color  = (60, 60, 220) if filled else (90, 90, 90)
        cx = w - pad - dot_r - (limit - 1 - i) * dot_gap
        cy = pad + dot_r
        cv2.circle(frame, (cx, cy), dot_r, color, -1 if filled else 1)

def draw_debug_overlay(frame, gaze, violations, elapsed_violation, calibrated):
    if not DEBUG_MODE:
        return
    h, w = frame.shape[:2]
    direction  = gaze.get('direction', '?') if gaze else '?'
    confidence = gaze.get('confidence', 0.0) if gaze else 0.0
    viol_short = violations[0][:24] if violations else "ok"
    cal_tag    = "CAL" if calibrated else "NO-CAL"
    line = f"{cal_tag} | gaze:{direction} {confidence:.2f} | {viol_short} | {elapsed_violation:.1f}s"
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, line, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 120), 1)

def draw_calibration_overlay(frame, seconds_left, face_detected, samples_count):
    """Pantalla durante la fase de calibración."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "CALIBRACION",
                (w // 2 - 180, h // 2 - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 220, 255), 3)
    cv2.putText(frame, "Mira al CENTRO con OJOS ABIERTOS",
                (w // 2 - 270, h // 2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    cv2.putText(frame, str(seconds_left),
                (w // 2 - 28, h // 2 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 220, 255), 5)

    status_text  = "Rostro OK" if face_detected else "No detecto tu rostro"
    status_color = (100, 220, 100) if face_detected else (80, 80, 240)
    cv2.putText(frame, status_text,
                (w // 2 - 110, h // 2 + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    # Muestras acumuladas (feedback visible)
    cv2.putText(frame, f"Muestras: {samples_count}",
                (w // 2 - 70, h // 2 + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

def draw_gaze_indicator(frame, direction):
    """Flecha grande arriba del frame indicando dirección detectada (cuando != center)."""
    if direction in (None, 'center', 'unknown'):
        return
    h, w = frame.shape[:2]
    label = direction.upper()
    cv2.putText(frame, f"<< MIRADA: {label} >>",
                (w // 2 - 180, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

def show_no_face_warning(frame):
    """Aviso discreto cuando no detecta rostro durante monitoreo."""
    h, w = frame.shape[:2]
    cv2.putText(frame, "Sin rostro detectado",
                (12, h - 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 240), 2)

def show_exam_closed(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cx, cy = w // 2, h // 2
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
        return

    cam = Camera()
    det = Detector(face_path, eye_path)
    ana = Analysis(violation_threshold_seconds=1.0, consecutive_frames=4)
    alt = Alerter()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    strikes          = 0
    exam_closed      = False
    last_strike_time = 0.0
    cal_start        = time.time()
    in_calibration   = True

    _init_window()
    print("Sistema iniciado. Calibrando 5s. Teclas: 'q'=salir, 'd'=debug, 'r'=recalibrar.")
    global DEBUG_MODE

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        proc = cv2.resize(frame, (PROCESS_W, PROCESS_H))
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)

        faces = det.detect_face(gray)
        pupils_and_rois = []
        all_eyes        = []

        for (x, y, fw, fh) in faces:
            cv2.rectangle(proc, (x, y), (x + fw, y + fh), (255, 180, 0), 2)
            roi_gray  = gray[y:y+fh, x:x+fw]
            roi_color = proc[y:y+fh, x:x+fw]
            eyes = det.detect_eyes(roi_gray)
            all_eyes.extend(eyes)
            for eye_idx, (ex, ey, ew, eh) in enumerate(eyes):
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 220, 0), 2)
                eye_roi  = roi_gray[ey:ey+eh, ex:ex+ew]
                pupil_raw = det.get_pupil_center(eye_roi, eye_index=eye_idx)
                pupil = det.smooth_pupil(pupil_raw, eye_idx, eye_roi.shape)
                pupils_and_rois.append((pupil, eye_roi.shape))
                if pupil:
                    px, py = pupil
                    cv2.circle(roi_color, (ex + px, ey + py), 4, (0, 0, 230), -1)

        # ── FASE 1: CALIBRACIÓN ──────────────────────────────────────────
        if in_calibration:
            det.add_calibration_sample(pupils_and_rois)
            elapsed = time.time() - cal_start
            seconds_left = max(1, int(CALIBRATION_SEC - elapsed) + 1)
            samples = det.calibration_samples_count()
            if elapsed >= CALIBRATION_SEC:
                ok = det.finish_calibration()
                in_calibration = False
                base = det.get_baseline()
                status = "OK" if ok else "INSUFICIENTE (usa umbrales fijos — presiona 'r' para reintentar)"
                print(f"[CALIBRACION] {status} | baseline: {base}")
            else:
                draw_calibration_overlay(proc, seconds_left, len(faces) > 0, samples)
                display = cv2.resize(proc, (WINDOW_W, WINDOW_H))
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

        # ── FASE 2: MONITOREO ────────────────────────────────────────────
        gaze_trigger, gaze = det.should_trigger_strike(pupils_and_rois)

        pupil_centers_only = [p for (p, _) in pupils_and_rois if p is not None]
        analysis_trigger, violations = ana.evaluate(faces, all_eyes, pupil_centers_only, proc.shape)

        elapsed_violation = time.time() - ana.last_clean_time if ana.is_violating else 0.0

        if len(faces) == 0:
            show_no_face_warning(proc)

        draw_gaze_indicator(proc, gaze.get('direction') if gaze else None)
        draw_debug_overlay(proc, gaze, violations, elapsed_violation, det.is_calibrated())
        draw_strikes(proc, strikes)

        triggered = gaze_trigger or analysis_trigger
        now = time.time()

        if triggered and (now - last_strike_time) >= STRIKE_COOLDOWN_SEC:
            strikes += 1
            last_strike_time = now
            if gaze_trigger:
                reason = f"MIRADA {gaze['direction'].upper()}"
            elif violations:
                reason = violations[0]
            else:
                reason = "análisis"
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

        display = cv2.resize(proc, (WINDOW_W, WINDOW_H))
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            DEBUG_MODE = not DEBUG_MODE
            print(f"[DEBUG] {'ON' if DEBUG_MODE else 'OFF'}")
        elif key == ord('r'):
            det.reset_calibration()
            cal_start = time.time()
            in_calibration = True
            print("[RECALIBRAR] Iniciando nueva calibración...")

    cam.release()
    cv2.destroyAllWindows()

    if exam_closed:
        print(f"\n[SISTEMA] Examen cerrado automáticamente por {LIMITE_OPORTUNIDADES} strikes.")

if __name__ == "__main__":
    main()
