import cv2

class Camera:
    def __init__(self, device_index=0):
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            # Intentar con índice 1 si el 0 falla (común en laptops con cámaras virtuales o múltiples)
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                raise Exception("No se pudo acceder a la cámara. Verifica que no esté siendo usada por otra app (Zoom, Teams, etc.) o prueba conectarla de nuevo.")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()
