import cv2
import platform
import os
import threading

if platform.system() == "Windows":
    import winsound

class Alerter:
    def __init__(self):
        self.os_type = platform.system()
        self._sound_lock = threading.Lock()  # evita solapar múltiples beeps

    def trigger_visual_alert(self, frame, violations):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 20)
        y_offset = 50
        for v in violations:
            cv2.putText(frame, f"ALERTA: {v}", (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 40

    def trigger_sound_alert(self):
        """Lanza el beep en un thread separado para no congelar el video."""
        t = threading.Thread(target=self._play_sound, daemon=True)
        t.start()

    def _play_sound(self):
        # Evitar solapamiento: si ya hay un beep sonando, descartar este
        if not self._sound_lock.acquire(blocking=False):
            return
        try:
            if self.os_type == "Windows":
                winsound.Beep(1000, 500)
            elif self.os_type == "Linux":
                print("\a", end='', flush=True)
            elif self.os_type == "Darwin":
                os.system('say "Alerta"')
            else:
                print("\007", end='', flush=True)
        finally:
            self._sound_lock.release()
