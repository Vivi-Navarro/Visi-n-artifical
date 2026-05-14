import cv2
import platform
import os
import threading
import time
from playsound import playsound

# Importar winsound solo si estamos en Windows
if platform.system() == "Windows":
    import winsound

class Alerter:
    def __init__(self):
        self.os_type = platform.system()
        self.video_path = os.path.join("data", "videos", "alerta.mp4")
        self.audio_path = os.path.join("data", "sounds", "alerta.mp3")

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

    def _play_audio_task(self):
        """
        Tarea para el hilo secundario que reproduce el sonido.
        """
        try:
            if os.path.exists(self.audio_path):
                playsound(os.path.abspath(self.audio_path))
        except Exception as e:
            print(f"Error al reproducir audio: {e}")

    def play_video_alert(self):
        """
        Reproduce el video de alerta y el audio sincronizados.
        Se cierra manualmente presionando 'q' o cuando termina el video.
        """
        if not os.path.exists(self.video_path):
            print(f"Aviso: No se encontro el video en {self.video_path}")
            return

        # Iniciar audio en un hilo separado
        audio_thread = threading.Thread(target=self._play_audio_task)
        audio_thread.daemon = True
        audio_thread.start()

        cap_video = cv2.VideoCapture(self.video_path)
        window_name = "ALERTA DE SEGURIDAD"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(window_name, 900, 100)

        while cap_video.isOpened():
            ret, vframe = cap_video.read()
            if not ret:
                break
            
            # Redimensionar para la ventana lateral
            height, width = vframe.shape[:2]
            if width > 500:
                vframe = cv2.resize(vframe, (500, int(height * (500 / width))))

            cv2.imshow(window_name, vframe)
            
            # Se cierra al presionar 'q'
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        cap_video.release()
        cv2.destroyWindow(window_name)
