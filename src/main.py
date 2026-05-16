import cv2
import os
from capture import Camera
from detection import Detector
from analysis import Analysis
from alerts import Alerter

LIMITE_OPORTUNIDADES = 5

def draw_strikes(frame, count, limit=LIMITE_OPORTUNIDADES):
    """Dibuja el contador de strikes en la esquina superior derecha."""
    h, w = frame.shape[:2]
    for i in range(limit):
        color = (0, 0, 255) if i < count else (200, 200, 200)
        cx = w - 30 - (limit - 1 - i) * 28
        cv2.circle(frame, (cx, 30), 10, color, -1)
    label = f"Strikes: {count}/{limit}"
    cv2.putText(frame, label, (w - 200, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def show_exam_closed(frame):
    """Muestra pantalla de cierre de examen."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    msg1 = "EXAMEN CERRADO"
    msg2 = "Se detectaron demasiadas violaciones"
    cv2.putText(frame, msg1,
                (frame.shape[1]//2 - 200, frame.shape[0]//2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    cv2.putText(frame, msg2,
                (frame.shape[1]//2 - 260, frame.shape[0]//2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Monitor de Examen - Vision Clasica", frame)
    cv2.waitKey(3000)  # Mostrar 3 segundos antes de cerrar

def main():
    face_path = os.path.join("data", "cascades", "haarcascade_frontalface_default.xml")
    eye_path  = os.path.join("data", "cascades", "haarcascade_eye.xml")

    if not os.path.exists(face_path) or not os.path.exists(eye_path):
        print("Error: No se encontraron los archivos Haar Cascade en data/cascades/")
        print("Asegurate de descargar haarcascade_frontalface_default.xml y haarcascade_eye.xml")
        return

    cam = Camera()
    det = Detector(face_path, eye_path)
    ana = Analysis(violation_threshold_seconds=2)
    alt = Alerter()

    strikes = 0  # Contador de strikes
    exam_closed = False

    print("Sistema iniciado. Presiona 'q' para salir.")

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = det.detect_face(gray)
        all_eyes = []
        pupil_centers = []

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray  = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = det.detect_eyes(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                all_eyes.append((ex, ey, ew, eh))
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

                eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                pupil = det.get_pupil_center(eye_roi)
                if pupil:
                    pupil_centers.append(pupil)
                    px, py = pupil
                    cv2.circle(roi_color, (ex + px, ey + py), 2, (0, 0, 255), -1)

        is_alarm, violations = ana.evaluate(faces, all_eyes, pupil_centers, frame.shape)

        if violations:
            alt.trigger_visual_alert(frame, violations)
            if is_alarm:
                alt.trigger_sound_alert()
                strikes += 1  # ← Incremento correcto en Python
                print(f"[ALERTA] Strike {strikes}/{LIMITE_OPORTUNIDADES}")

                if strikes >= LIMITE_OPORTUNIDADES:
                    exam_closed = True
                    show_exam_closed(frame)
                    break  # Cierra el examen

        # Mostrar strikes en pantalla en cada frame
        draw_strikes(frame, strikes)

        cv2.imshow("Monitor de Examen - Vision Clasica", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    if exam_closed:
        print(f"\n[SISTEMA] Examen cerrado automáticamente por {LIMITE_OPORTUNIDADES} strikes.")

if __name__ == "__main__":
    main()