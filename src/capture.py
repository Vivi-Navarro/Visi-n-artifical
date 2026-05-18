# Este archivo se encarga de manejar la cámara

import cv2

class Camera:
    # Clase que controla la cámara web

    def __init__(self, device_index=0):
        # Intentamos abrir la cámara (por defecto la 0, que es la principal)
        self.cap = cv2.VideoCapture(device_index)

        if not self.cap.isOpened():
            # Si no abrió, probamos con la cámara 1 (por si hay varias)
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                # Si tampoco funciona, avisamos al usuario
                raise Exception("No se pudo acceder a la cámara. Verifica que no esté siendo usada por otra app (Zoom, Teams, etc.) o prueba conectarla de nuevo.")

    def get_frame(self):
        # Captura un frame (una foto) de la cámara y lo devuelve
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        # Libera la cámara para que otras apps la puedan usar
        self.cap.release()
