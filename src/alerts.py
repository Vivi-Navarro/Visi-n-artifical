import cv2
import platform
import os

# Importar winsound solo si estamos en Windows
if platform.system() == "Windows":
    import winsound

class Alerter:
    def __init__(self):
        self.os_type = platform.system()

    def trigger_visual_alert(self, frame, violations):
        # Marco rojo en el borde del frame
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 20)
        
        y_offset = 50
        for v in violations:
            cv2.putText(frame, f"ALERTA: {v}", (50, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 40

    def trigger_sound_alert(self):
        """
        Ejecuta la alarma sonora dependiendo del sistema operativo.
        """
        if self.os_type == "Windows":
            # Pitido estándar de Windows
            winsound.Beep(1000, 500)
        elif self.os_type == "Linux":
            # Sonido de campana de sistema (PC Speaker)
            # En terminales Linux, esto genera un 'beep'
            print("\a", end='', flush=True)
        elif self.os_type == "Darwin": # macOS
            # Usa el sintetizador de voz integrado de Mac
            os.system('say "Alerta"')
        else:
            # Fallback para otros sistemas
            print("\007", end='', flush=True)
