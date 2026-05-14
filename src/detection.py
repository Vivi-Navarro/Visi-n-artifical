import cv2
import numpy as np

class Detector:
    def __init__(self, face_cascade_path, eye_cascade_path):
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    def detect_face(self, gray_frame):
        faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        return faces

    def detect_eyes(self, gray_face_roi):
        eyes = self.eye_cascade.detectMultiScale(gray_face_roi, 1.1, 10)
        return eyes

    def get_pupil_center(self, eye_gray_roi):
        """
        Detecta el centro de la pupila usando umbralización y contornos.
        """
        # Suavizado para reducir ruido
        eye_gray_roi = cv2.GaussianBlur(eye_gray_roi, (7, 7), 0)
        
        # Umbralización inversa para resaltar la pupila (parte más oscura)
        _, threshold = cv2.threshold(eye_gray_roi, 30, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        if len(contours) > 0:
            M = cv2.moments(contours[0])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy)
        return None
