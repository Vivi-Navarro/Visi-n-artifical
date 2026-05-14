import cv2
import platform
import os
import time

# Importar winsound solo si estamos en Windows
if platform.system() == "Windows":
    import winsound

class Alerter:
    def __init__(self):
        self.os_type = platform.system()
        self.video_path = os.path.join("data", "videos", "alerta.mp4")

    def trigger_visual_alert(self, frame, violations):
        # Marco rojo en el borde del frame
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 20)
        
        y_offset = 50
        for v in violations:
            cv2.putText(frame, f"ALERTA: {v}", (50, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 40

    def trigger_sound_alert(self):
        if self.os_type == "Windows":
            winsound.Beep(1000, 500)
        elif self.os_type == "Linux":
            print("\a", end='', flush=True)
        elif self.os_type == "Darwin":
            os.system('say "Alerta"')

    def play_video_alert(self):
        """
        Reproduce el video de alerta en una ventana independiente.
        """
        if not os.path.exists(self.video_path):
            print(f"Aviso: No se encontro el video en {self.video_path}")
            return

        cap_video = cv2.VideoCapture(self.video_path)
        cv2.namedWindow("ALERTA DE SEGURIDAD", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("ALERTA DE SEGURIDAD", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while cap_video.isOpened():
            ret, vframe = cap_video.read()
            if not ret:
                break
            
            cv2.imshow("ALERTA DE SEGURIDAD", vframe)
            # Reproducir a velocidad normal (aprox 30fps)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        cap_video.release()
        cv2.destroyWindow("ALERTA DE SEGURIDAD")
