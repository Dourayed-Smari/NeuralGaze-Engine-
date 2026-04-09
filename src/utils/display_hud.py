import cv2

FACE_CONNECTIONS = [
    (10, 151), (151, 9), (9, 10),
    (234, 127), (127, 162), (162, 21), (21, 54),
    (454, 356), (356, 389), (389, 251), (251, 284),
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 320), (320, 375), (375, 291)
]

EYE_CONNECTIONS = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153),
    (362, 382), (382, 381), (381, 380), (380, 374), (374, 373)
]

def draw_enhanced_landmarks(frame, landmarks, facial_axis_points, left_iris, right_iris, pose_landmarks=None, mp_pose=None):
    h, w = frame.shape[:2]
    for start_idx, end_idx in FACE_CONNECTIONS:
        start_p = (int(landmarks.landmark[start_idx].x * w), int(landmarks.landmark[start_idx].y * h))
        end_p = (int(landmarks.landmark[end_idx].x * w), int(landmarks.landmark[end_idx].y * h))
        cv2.line(frame, start_p, end_p, (0, 255, 255), 1)
    
    for start_idx, end_idx in EYE_CONNECTIONS:
        start_p = (int(landmarks.landmark[start_idx].x * w), int(landmarks.landmark[start_idx].y * h))
        end_p = (int(landmarks.landmark[end_idx].x * w), int(landmarks.landmark[end_idx].y * h))
        cv2.line(frame, start_p, end_p, (0, 255, 0), 2)
    
    for idx in left_iris + right_iris:
        cv2.circle(frame, (int(landmarks.landmark[idx].x * w), int(landmarks.landmark[idx].y * h)), 2, (255, 0, 0), -1)
    
    for _, idx in facial_axis_points.items():
        cv2.circle(frame, (int(landmarks.landmark[idx].x * w), int(landmarks.landmark[idx].y * h)), 3, (255, 255, 0), -1)
    
    if pose_landmarks and mp_pose:
        points = [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR, 
                  mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]
        for p_idx in points:
            try: cv2.circle(frame, (int(pose_landmarks.landmark[p_idx].x * w), int(pose_landmarks.landmark[p_idx].y * h)), 4, (255, 165, 0), -1)
            except: pass

def show_calibration_screen(frame, calibration_samples, calibration_threshold, dynamic_mode, pose_landmarks=None):
    display_frame = cv2.flip(frame, 1)
    h, w = display_frame.shape[:2]
    cv2.circle(display_frame, (w//2, h//2), 25, (0, 255, 0), -1)
    cv2.circle(display_frame, (w//2, h//2), 30, (255, 255, 255), 3)
    progress = len(calibration_samples) / calibration_threshold
    cv2.rectangle(display_frame, (50, h - 50), (int(50 + progress * 300), h - 30), (0, 255, 0), -1)
    cv2.rectangle(display_frame, (50, h - 50), (350, h - 30), (255, 255, 255), 2)
    cv2.putText(display_frame, f"CALIBRATION: LOOK AT CENTER AND PRESS [C] ({len(calibration_samples)}/{calibration_threshold})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('NeuralGaze Engine', display_frame)

def show_tracking(frame, coords, left_bounds, right_bounds, face_h, face_v, tracking_confidence, dynamic_mode, pose_landmarks=None):
    display_frame = cv2.flip(frame, 1)
    h, w = display_frame.shape[:2]
    if left_bounds and right_bounds:
        lx, ly, lw, lh = left_bounds
        rx, ry, rw, rh = right_bounds
        cv2.rectangle(display_frame, (w - lx - lw, ly), (w - lx, ly + lh), (0, 255, 0), 2)
        cv2.rectangle(display_frame, (w - rx - rw, ry), (w - rx, ry + rh), (0, 255, 0), 2)
    if coords:
        gx, gy = int((1 - coords[0]) * w), int(coords[1] * h)
        cv2.circle(display_frame, (gx, gy), 8 if dynamic_mode else 10, (0, 255, 0) if dynamic_mode else (0, 165, 255), -1)
    mode_text = "MODE: FAST MOUSE" if dynamic_mode else "MODE: SLOW MOUSE"
    cv2.putText(display_frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if dynamic_mode else (0, 165, 255), 2)
    cv2.putText(display_frame, f"Signal Quality: {tracking_confidence:.2f}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow('NeuralGaze Engine', display_frame)

def show_error_frame(frame, message):
    display_frame = cv2.flip(frame, 1)
    cv2.putText(display_frame, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('NeuralGaze Engine', display_frame)
