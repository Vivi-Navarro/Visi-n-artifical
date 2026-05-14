import cv2
import winsound

class Alerter:
    def __init__(self):
        self.alarm_playing = False

    def trigger_visual_alert(self, frame, violations):
        # Marco rojo
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[2] if len(frame.shape)>2 else frame.shape[0]), (0, 0, 255), 20)
        
        y_offset = 50
        for v in violations:
            cv2.putText(frame, f"ALERTA: {v}", (50, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 40

    def trigger_sound_alert(self):
        # Pitido de 1000Hz por 500ms
        winsound.Beep(1000, 500)
