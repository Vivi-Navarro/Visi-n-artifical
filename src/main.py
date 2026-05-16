import cv2
import os
from capture import Camera
from detection import Detector
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
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "EXAMEN CERRADO",
                (frame.shape[1]//2 - 200, frame.shape[0]//2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    cv2.putText(frame, "Se detectaron demasiadas violaciones",
                (frame.shape[1]//2 - 260, frame.shape[0]//2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Monitor de Examen - Vision Clasica", frame)
    cv2.waitKey(3000)

def main():
    face_path = os.path.join("data", "cascades", "haarcascade_frontalface_default.xml")
    eye_path  = os.path.join("data", "cascades", "haarcascade_eye.xml")

    if not os.path.exists(face_path) or not os.path.exists(eye_path):
        print("Error: No se encontraron los archivos Haar Cascade en data/cascades/")
        print("Asegurate de descargar haarcascade_frontalface_default.xml y haarcascade_eye.xml")
        return

    cam = Camera()
    det = Detector(face_path, eye_path)
    alt = Alerter()

    strikes = 0
    exam_closed = False

    print("Sistema iniciado. Presiona 'q' para salir.")

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = det.detect_face(gray)
        pupils_and_rois = []

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray  = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = det.detect_eyes(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

                eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                pupil = det.get_pupil_center(eye_roi)
                pupils_and_rois.append((pupil, eye_roi.shape))

                if pupil:
                    px, py = pupil
                    cv2.circle(roi_color, (ex + px, ey + py), 2, (0, 0, 255), -1)

        # Evaluación una sola vez por frame, con filtro anti-parpadeo
        trigger, gaze = det.should_trigger_strike(pupils_and_rois)

        if trigger:
            strikes += 1
            print(f"[STRIKE {strikes}/{LIMITE_OPORTUNIDADES}] Mirando a la {gaze['direction']}")
            alt.trigger_sound_alert()

            if strikes >= LIMITE_OPORTUNIDADES:
                exam_closed = True
                show_exam_closed(frame)
                break

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