import cv2
import os
from capture import Camera
from detection import Detector
from analysis import Analysis
from alerts import Alerter

def main():
    # Rutas de los clasificadores
    face_path = os.path.join("data", "cascades", "haarcascade_frontalface_default.xml")
    eye_path = os.path.join("data", "cascades", "haarcascade_eye.xml")

    # Verificar existencia de archivos Haar Cascade
    if not os.path.exists(face_path) or not os.path.exists(eye_path):
        print("Error: No se encontraron los archivos Haar Cascade en data/cascades/")
        print("Asegurate de descargar haarcascade_frontalface_default.xml y haarcascade_eye.xml")
        return

    cam = Camera()
    det = Detector(face_path, eye_path)
    ana = Analysis(violation_threshold_seconds=2)
    alt = Alerter()

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
            roi_gray = gray[y:y+h, x:x+w]
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
                alt.play_video_alert() # Se reproduce el video cuando se confirma la trampa

        cv2.imshow("Monitor de Examen - Vision Clasica", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
