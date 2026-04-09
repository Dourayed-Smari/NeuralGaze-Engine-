import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from utils.math_utils import *
from utils.display_hud import *

class NeuralVisionEngine:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cv2.namedWindow('NeuralGaze Engine', cv2.WINDOW_NORMAL)
        try: cv2.setWindowProperty('NeuralGaze Engine', cv2.WND_PROP_TOPMOST, 1)
        except: pass
        self.gaze_history, self.pupil_history, self.face_axis_history = deque(maxlen=10), deque(maxlen=3), deque(maxlen=3)
        self.calibrated, self.center_point, self.calibration_samples, self.calibration_threshold = False, None, [], 8
        self.show_landmarks, self.dynamic_mode, self.pose_weight = True, True, 0.15
        self.tracking_confidence, self.eye_openness_threshold = 0.0, 0.34
        self.blink_start_time, self.is_blinking = None, False
        self.prev_left_pupil, self.prev_right_pupil, self.fallback_iris_coords = None, None, None

    def get_gaze(self):
        ret, frame = self.cap.read()
        if not ret: return None, None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        if not face_results.multi_face_landmarks:
            show_error_frame(frame, "No face detected")
            return None, None
        landmarks = face_results.multi_face_landmarks[0]
        pose_landmarks = pose_results.pose_landmarks if pose_results.pose_landmarks else None
        if self.show_landmarks: draw_enhanced_landmarks(frame, landmarks, FACIAL_AXIS_POINTS, LEFT_IRIS, RIGHT_IRIS, pose_landmarks, self.mp_pose)
        
        left_op, right_op = get_eye_openness(landmarks, LEFT_EYE), get_eye_openness(landmarks, RIGHT_EYE)
        click_type = None
        if left_op < self.eye_openness_threshold and right_op < self.eye_openness_threshold:
            if not self.is_blinking: self.is_blinking, self.blink_start_time = True, time.time()
        else:
            if self.is_blinking:
                self.is_blinking = False
                if self.blink_start_time:
                    dur = time.time() - self.blink_start_time
                    if 0.15 < dur < 1.0: click_type = "left"
                    elif 1.0 <= dur < 3.0: click_type = "right"
                    
        if left_op < self.eye_openness_threshold or right_op < self.eye_openness_threshold:
            if self.prev_left_pupil and self.prev_right_pupil: left_pupil, right_pupil = self.prev_left_pupil, self.prev_right_pupil
            else: return None, None
        else:
            _, lb = extract_eye_region(frame, landmarks, LEFT_EYE)
            _, rb = extract_eye_region(frame, landmarks, RIGHT_EYE)
            if not lb or not rb:
                if self.prev_left_pupil and self.prev_right_pupil: left_pupil, right_pupil = self.prev_left_pupil, self.prev_right_pupil
                else: return None, None
            else:
                left_pupil, self.fallback_iris_coords = get_anatomical_pupil_center(frame, landmarks, LEFT_IRIS, lb, self.fallback_iris_coords)
                right_pupil, _ = get_anatomical_pupil_center(frame, landmarks, RIGHT_IRIS, rb, self.fallback_iris_coords)
                if not left_pupil or not right_pupil:
                    if self.prev_left_pupil and self.prev_right_pupil: left_pupil, right_pupil = self.prev_left_pupil, self.prev_right_pupil
                    else: return None, None
                else: self.prev_left_pupil, self.prev_right_pupil = left_pupil, right_pupil

        self.pupil_history.append((left_pupil, right_pupil))
        avg = np.mean(self.pupil_history, axis=0)
        lp, rp = avg[0], avg[1]
        fh, fv = get_facial_axis_direction(landmarks)
        if pose_landmarks:
            ph, pv = get_pose_direction(pose_landmarks)
            ch, cv_offset = combine_face_and_pose(fh, fv, ph, pv, self.pose_weight, self.face_axis_history)
        else: ch, cv_offset = fh, fv
        
        egx, egy = (lp[0]+rp[0])/2, (lp[1]+rp[1])/2
        sens_x, sens_y = (1.7, 2.3) if self.dynamic_mode else (1.4, 2.0)
        cx, cy = egx + (ch * (0.28 if self.dynamic_mode else 0.22)), egy + (cv_offset * (0.35 if self.dynamic_mode else 0.28))
        cx, cy = np.clip(cx, 0, 1), np.clip(cy, 0, 1)
        
        if not self.calibrated:
            show_calibration_screen(frame, self.calibration_samples, self.calibration_threshold, self.dynamic_mode)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                self.calibration_samples.append((cx, cy))
                if len(self.calibration_samples) >= self.calibration_threshold:
                    self.center_point = np.mean(self.calibration_samples, axis=0)
                    self.calibrated = True
            return None, None
        
        if self.center_point is not None:
            cx = np.clip(0.5 + (cx - self.center_point[0]) * sens_x, 0, 1)
            cy = np.clip(0.5 + (cy - self.center_point[1]) * sens_y, 0, 1)
        
        self.gaze_history.append((cx, cy))
        weights = np.linspace(0.2, 1.0, len(self.gaze_history))
        weights /= weights.sum()
        sc = np.sum([np.array(c) * w for c, w in zip(self.gaze_history, weights)], axis=0)
        
        if len(self.gaze_history) >= 3:
            self.tracking_confidence = max(0, 1 - np.var(self.gaze_history, axis=0).sum() * 10)
        
        show_tracking(frame, sc, lb if 'lb' in locals() else None, rb if 'rb' in locals() else None, fh, fv, self.tracking_confidence, self.dynamic_mode, pose_landmarks)
        return sc, click_type

    def should_quit(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): return True
        if key == ord('m'): self.dynamic_mode = not self.dynamic_mode
        if key == ord('l'): self.show_landmarks = not self.show_landmarks
        return False

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
