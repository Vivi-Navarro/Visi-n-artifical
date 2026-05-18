# Este es el archivo principal - aquí se ejecuta todo el programa
# Conecta la cámara, la detección, el análisis y las alertas

import cv2
import os
import sys
import time
from capture import Camera
from detection import Detector
from analysis import Analysis
from alerts import Alerter

# Arreglamos problemas de caracteres especiales en la consola de Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ============================
# CONFIGURACIÓN GENERAL
# ============================
LIMITE_OPORTUNIDADES = 5       # Strikes máximos antes de cerrar el examen
DEBUG_MODE           = True    # Mostrar info de debug (se puede cambiar con 'd')
STRIKE_COOLDOWN_SEC  = 3.0     # Segundos mínimos entre cada strike
CALIBRATION_SEC      = 5       # Duración de la calibración en segundos

# Configuración de la ventana de video
WINDOW_NAME   = "Monitor"
WINDOW_W      = 800
WINDOW_H      = 600
WINDOW_POS_X  = None           # None = esquina derecha automática
WINDOW_POS_Y  = 20
PROCESS_W     = 640            # Resolución a la que procesamos (no es la ventana)
PROCESS_H     = 480

# FUNCIONES DE DIBUJO

def _get_screen_width():
    # Obtiene el ancho de la pantalla para posicionar la ventana
    try:
        import ctypes
        return int(ctypes.windll.user32.GetSystemMetrics(0))
    except Exception:
        return 1500

def _init_window():
    # Crea y posiciona la ventana de video
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)
    screen_w = _get_screen_width()
    pos_x = WINDOW_POS_X if WINDOW_POS_X is not None else screen_w - WINDOW_W - 10
    cv2.moveWindow(WINDOW_NAME, pos_x, WINDOW_POS_Y)

def draw_strikes(frame, count, limit=LIMITE_OPORTUNIDADES):
    # Dibuja los puntos de strikes en la esquina superior derecha
    # Los rojos son los que ya llevas, los grises los que te quedan
    h, w = frame.shape[:2]
    dot_r, dot_gap, pad = 8, 22, 14
    for i in range(limit):
        filled = i < count
        color  = (60, 60, 220) if filled else (90, 90, 90)
        cx = w - pad - dot_r - (limit - 1 - i) * dot_gap
        cy = pad + dot_r
        cv2.circle(frame, (cx, cy), dot_r, color, -1 if filled else 1)

def draw_debug_overlay(frame, gaze, violations, elapsed_violation, calibrated):
    # Dibuja la barra de debug abajo con información técnica
    if not DEBUG_MODE:
        return
    h, w = frame.shape[:2]
    direction  = gaze.get('direction', '?') if gaze else '?'
    confidence = gaze.get('confidence', 0.0) if gaze else 0.0
    viol_short = violations[0][:24] if violations else "ok"
    cal_tag    = "CAL" if calibrated else "NO-CAL"
    line = f"{cal_tag} | gaze:{direction} {confidence:.2f} | {viol_short} | {elapsed_violation:.1f}s"
    # Fondo semitransparente para que se lea bien
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, line, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 120), 1)

def draw_calibration_overlay(frame, seconds_left, face_detected, samples_count):
    # Pantalla que se muestra durante la calibración
    h, w = frame.shape[:2]
    # Fondo oscuro semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Título "CALIBRACION"
    cv2.putText(frame, "CALIBRACION",
                (w // 2 - 180, h // 2 - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 220, 255), 3)
    # Instrucción para el usuario
    cv2.putText(frame, "Mira al CENTRO con OJOS ABIERTOS",
                (w // 2 - 270, h // 2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    # Cuenta regresiva grande
    cv2.putText(frame, str(seconds_left),
                (w // 2 - 28, h // 2 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 220, 255), 5)

    # Estado del rostro (si lo detecta o no)
    status_text  = "Rostro OK" if face_detected else "No detecto tu rostro"
    status_color = (100, 220, 100) if face_detected else (80, 80, 240)
    cv2.putText(frame, status_text,
                (w // 2 - 110, h // 2 + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    # Cuántas muestras llevamos
    cv2.putText(frame, f"Muestras: {samples_count}",
                (w // 2 - 70, h // 2 + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

def draw_gaze_indicator(frame, direction):
    # Muestra un texto arriba indicando hacia dónde estás mirando
    if direction in (None, 'center', 'unknown'):
        return
    h, w = frame.shape[:2]
    label = direction.upper()
    cv2.putText(frame, f"<< MIRADA: {label} >>",
                (w // 2 - 180, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

def show_no_face_warning(frame):
    # Aviso discreto cuando no detecta tu cara
    h, w = frame.shape[:2]
    cv2.putText(frame, "Sin rostro detectado",
                (12, h - 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 240), 2)

def show_exam_closed(frame):
    # Pantalla final cuando se cierra el examen por demasiados strikes
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


# FUNCIÓN PRINCIPAL

def main():
    # Rutas a los archivos Haar Cascade (los necesita OpenCV para detectar)
    face_path = os.path.join("data", "cascades", "haarcascade_frontalface_default.xml")
    eye_path  = os.path.join("data", "cascades", "haarcascade_eye.xml")
    if not os.path.exists(face_path) or not os.path.exists(eye_path):
        print("Error: No se encontraron los archivos Haar Cascade en data/cascades/")
        return

    # Creamos todos los objetos principales
    cam = Camera()                 # Cámara
    det = Detector(face_path, eye_path)  # Detector de caras/ojos/pupilas
    ana = Analysis(violation_threshold_seconds=1.0, consecutive_frames=4)  # Analizador
    alt = Alerter()                # Sistema de alertas
    # CLAHE mejora la iluminación de la imagen
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Variables de estado del programa
    strikes          = 0           # Strikes acumulados
    exam_closed      = False       # ¿Se cerró el examen?
    last_strike_time = 0.0         # Cuándo fue el último strike
    cal_start        = time.time() # Cuándo empezó la calibración
    in_calibration   = True        # ¿Estamos calibrando?

    _init_window()
    print("Sistema iniciado. Calibrando 5s. Teclas: 'q'=salir, 'd'=debug, 'r'=recalibrar.")
    global DEBUG_MODE

    # ============================
    # LOOP PRINCIPAL (se repite cada frame)
    # ============================
    while True:
        # Capturamos un frame de la cámara
        frame = cam.get_frame()
        if frame is None:
            break

        # Redimensionamos y convertimos a escala de grises
        proc = cv2.resize(frame, (PROCESS_W, PROCESS_H))
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        # Mejoramos la iluminación con CLAHE
        gray = clahe.apply(gray)

        # Detectamos caras en la imagen
        faces = det.detect_face(gray)
        pupils_and_rois = []
        all_eyes        = []

        # Para cada cara encontrada...
        for (x, y, fw, fh) in faces:
            # Dibujamos un rectángulo alrededor de la cara
            cv2.rectangle(proc, (x, y), (x + fw, y + fh), (255, 180, 0), 2)
            # Recortamos la zona de la cara
            roi_gray  = gray[y:y+fh, x:x+fw]
            roi_color = proc[y:y+fh, x:x+fw]
            # Buscamos ojos dentro de la cara
            eyes = det.detect_eyes(roi_gray)
            all_eyes.extend(eyes)
            # Para cada ojo encontrado...
            for eye_idx, (ex, ey, ew, eh) in enumerate(eyes):
                # Dibujamos un rectángulo alrededor del ojo
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 220, 0), 2)
                # Recortamos la zona del ojo
                eye_roi  = roi_gray[ey:ey+eh, ex:ex+ew]
                # Buscamos la pupila
                pupil_raw = det.get_pupil_center(eye_roi, eye_index=eye_idx)
                # Suavizamos la posición de la pupila
                pupil = det.smooth_pupil(pupil_raw, eye_idx, eye_roi.shape)
                pupils_and_rois.append((pupil, eye_roi.shape))
                if pupil:
                    # Dibujamos un punto rojo donde está la pupila
                    px, py = pupil
                    cv2.circle(roi_color, (ex + px, ey + py), 4, (0, 0, 230), -1)

        # ============================
        # FASE 1: CALIBRACIÓN
        # ============================
        if in_calibration:
            # Guardamos muestras de dónde están las pupilas
            det.add_calibration_sample(pupils_and_rois)
            elapsed = time.time() - cal_start
            seconds_left = max(1, int(CALIBRATION_SEC - elapsed) + 1)
            samples = det.calibration_samples_count()
            if elapsed >= CALIBRATION_SEC:
                # Terminó la calibración
                ok = det.finish_calibration()
                in_calibration = False
                base = det.get_baseline()
                status = "OK" if ok else "INSUFICIENTE (usa umbrales fijos — presiona 'r' para reintentar)"
                print(f"[CALIBRACION] {status} | baseline: {base}")
            else:
                # Aún calibrando: mostramos la pantalla de calibración
                draw_calibration_overlay(proc, seconds_left, len(faces) > 0, samples)
                display = cv2.resize(proc, (WINDOW_W, WINDOW_H))
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

        # ============================
        # FASE 2: MONITOREO
        # ============================
        # Revisamos si la mirada es sospechosa
        gaze_trigger, gaze = det.should_trigger_strike(pupils_and_rois)

        # Revisamos otras violaciones (cara fuera, giro, ausencia, etc.)
        pupil_centers_only = [p for (p, _) in pupils_and_rois if p is not None]
        analysis_trigger, violations = ana.evaluate(faces, all_eyes, pupil_centers_only, proc.shape)

        # Cuánto tiempo lleva la violación actual
        elapsed_violation = time.time() - ana.last_clean_time if ana.is_violating else 0.0

        # Si no hay cara, mostramos un aviso
        if len(faces) == 0:
            show_no_face_warning(proc)

        # Dibujamos los indicadores en pantalla
        draw_gaze_indicator(proc, gaze.get('direction') if gaze else None)
        draw_debug_overlay(proc, gaze, violations, elapsed_violation, det.is_calibrated())
        draw_strikes(proc, strikes)

        # ¿Hay que dar un strike?
        triggered = gaze_trigger or analysis_trigger
        now = time.time()

        if triggered and (now - last_strike_time) >= STRIKE_COOLDOWN_SEC:
            strikes += 1
            last_strike_time = now
            # Determinamos la razón del strike
            if gaze_trigger:
                reason = f"MIRADA {gaze['direction'].upper()}"
            elif violations:
                reason = violations[0]
            else:
                reason = "análisis"
            print(f"[STRIKE {strikes}/{LIMITE_OPORTUNIDADES}] Razón: {reason}")
            # Sonido de alerta
            alt.trigger_sound_alert()

            # Alerta visual en pantalla
            if violations:
                alt.trigger_visual_alert(proc, violations[:2])
            elif gaze_trigger:
                alt.trigger_visual_alert(proc, [f"GAZE: {gaze['direction'].upper()}"])

            # Si llegamos al límite de strikes, cerramos el examen
            if strikes >= LIMITE_OPORTUNIDADES:
                exam_closed = True
                show_exam_closed(proc)
                break

        # Mostramos el frame en la ventana
        display = cv2.resize(proc, (WINDOW_W, WINDOW_H))
        cv2.imshow(WINDOW_NAME, display)

        # Leemos teclas del usuario
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # 'q' = salir
            break
        elif key == ord('d'):
            # 'd' = activar/desactivar debug
            DEBUG_MODE = not DEBUG_MODE
            print(f"[DEBUG] {'ON' if DEBUG_MODE else 'OFF'}")
        elif key == ord('r'):
            # 'r' = recalibrar sin reiniciar el programa
            det.reset_calibration()
            cal_start = time.time()
            in_calibration = True
            print("[RECALIBRAR] Iniciando nueva calibración...")

    # Limpieza al salir
    cam.release()
    cv2.destroyAllWindows()

    if exam_closed:
        print(f"\n[SISTEMA] Examen cerrado automáticamente por {LIMITE_OPORTUNIDADES} strikes.")

if __name__ == "__main__":
    main()
