import cv2
import numpy as np
import time

class Detector:
    def __init__(self, face_cascade_path, eye_cascade_path):
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade  = cv2.CascadeClassifier(eye_cascade_path)

        self._gaze_off_since          = None
        self._last_blink_time         = None
        self._blink_tolerance_sec     = 0.5
        self._suspicious_threshold_sec = 2.5

    def detect_face(self, gray_frame):
        return self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    def detect_eyes(self, gray_face_roi):
        eyes = self.eye_cascade.detectMultiScale(
            gray_face_roi,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(25, 25),
            maxSize=(150, 150)
        )
        h = gray_face_roi.shape[0]
        eyes = [e for e in eyes if e[1] < h // 2]
        return eyes

    def get_pupil_center(self, eye_gray_roi):
        eye_gray_roi = cv2.GaussianBlur(eye_gray_roi, (7, 7), 0)

        threshold_val = int(np.percentile(eye_gray_roi, 20))
        threshold_val = min(threshold_val, 80)

        _, threshold = cv2.threshold(eye_gray_roi, threshold_val, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        h, w = eye_gray_roi.shape
        min_area = (h * w) * 0.02
        max_area = (h * w) * 0.50

        for contour in contours:
            area = cv2.contourArea(contour)
            if not (min_area < area < max_area):
                continue
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if 3 < cx < w - 3 and 3 < cy < h - 3:
                return (cx, cy)
        return None

    def get_gaze_direction(self, pupil_center, eye_roi_shape):
        if pupil_center is None:
            return None
        px, _ = pupil_center
        ratio = px / eye_roi_shape[1]
        if ratio < 0.30:
            return "left"
        elif ratio > 0.70:
            return "right"
        return "center"

    def analyze_gaze(self, pupils_and_rois):
        directions = []
        for p, s in pupils_and_rois:
            d = self.get_gaze_direction(p, s)
            if d is not None:
                directions.append(d)

        if not directions:
            return {'is_suspicious': False, 'direction': 'unknown', 'confidence': 0.0}

        off_center = [d for d in directions if d != 'center']
        confidence = len(off_center) / len(directions)

        if not off_center:
            return {'is_suspicious': False, 'direction': 'center', 'confidence': 0.0}

        left_count  = off_center.count('left')
        right_count = off_center.count('right')
        dominant = 'left' if left_count >= right_count else 'right'

        return {
            'is_suspicious': confidence >= 0.6,
            'direction': dominant,
            'confidence': confidence
        }

    def should_trigger_strike(self, pupils_and_rois):
        gaze = self.analyze_gaze(pupils_and_rois)
        now  = time.time()

        if gaze['direction'] == 'unknown':
            self._last_blink_time = now
            return False, gaze

        if self._last_blink_time is not None:
            if now - self._last_blink_time < self._blink_tolerance_sec:
                return False, gaze

        if not gaze['is_suspicious']:
            self._gaze_off_since = None
            return False, gaze

        if self._gaze_off_since is None:
            self._gaze_off_since = now

        elapsed = now - self._gaze_off_since

        if elapsed >= self._suspicious_threshold_sec:
            self._gaze_off_since = None
            return True, gaze

        return False, gaze