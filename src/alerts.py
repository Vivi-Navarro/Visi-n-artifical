# Este archivo se encarga de las alertas (visuales y sonoras)

import cv2
import platform
import os
import threading

# Solo importamos winsound si estamos en Windows (en otros SO no existe)
if platform.system() == "Windows":
    import winsound

class Alerter:
    # Clase que maneja las alertas cuando se detecta trampa

    def __init__(self):
        # Guardamos qué sistema operativo tenemos
        self.os_type = platform.system()
        # Candado para que no se encimen varios sonidos a la vez
        self._sound_lock = threading.Lock()

    def trigger_visual_alert(self, frame, violations):
        # Dibuja un marco rojo grueso alrededor de la pantalla
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 20)
        # Escribe cada violación como texto rojo en la imagen
        y_offset = 50
        for v in violations:
            cv2.putText(frame, f"ALERTA: {v}", (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 40

    def trigger_sound_alert(self):
        # Lanza el sonido en un hilo aparte para no congelar el video
        t = threading.Thread(target=self._play_sound, daemon=True)
        t.start()

    def _play_sound(self):
        # Si ya hay un sonido sonando, no lanzar otro encima
        if not self._sound_lock.acquire(blocking=False):
            return
        try:
            if self.os_type == "Windows":
                # Beep de 1000 Hz durante 500 ms
                winsound.Beep(1000, 500)
            elif self.os_type == "Linux":
                print("\a", end='', flush=True)
            elif self.os_type == "Darwin":
                # En Mac usa el comando "say" para hablar
                os.system('say "Alerta"')
            else:
                print("\007", end='', flush=True)
        finally:
            # Liberamos el candado para que pueda sonar otro después
            self._sound_lock.release()
