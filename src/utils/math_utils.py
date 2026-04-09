import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

# Constantes Physiologiques
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

FACIAL_AXIS_POINTS = {
    'nose_tip': 1, 'nose_bridge': 6, 'nose_bottom': 2,
    'forehead_center': 9, 'forehead_top': 10,
    'chin_tip': 175, 'chin_bottom': 18,
    'left_temple': 21, 'right_temple': 251,
    'left_cheek_center': 116, 'right_cheek_center': 345,
    'left_jaw': 172, 'right_jaw': 397,
    'left_eyebrow_inner': 70, 'right_eyebrow_inner': 300,
    'left_eyebrow_outer': 46, 'right_eyebrow_outer': 276,
    'left_eye_center': 468, 'right_eye_center': 473,
    'mouth_center': 13, 'upper_lip': 12, 'lower_lip': 15
}

def extract_eye_region(frame, landmarks, eye_indices):
    h, w = frame.shape[:2]
    points = []
    for idx in eye_indices:
        landmark = landmarks.landmark[idx]
        x, y = int(landmark.x * w), int(landmark.y * h)
        points.append((x, y))
    
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x_min, x_max = max(0, min(xs) - 35), min(w, max(xs) + 35)
    y_min, y_max = max(0, min(ys) - 25), min(h, max(ys) + 25)
    
    if x_max <= x_min or y_max <= y_min: return None, None
    return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max - x_min, y_max - y_min)

def get_anatomical_pupil_center(frame, landmarks, iris_indices, eye_bounds, fallback_iris_coords=None):
    if eye_bounds is None: return None, fallback_iris_coords
    h, w = frame.shape[:2]
    eye_x, eye_y, eye_w, eye_h = eye_bounds
    
    iris_points = []
    for idx in iris_indices:
        landmark = landmarks.landmark[idx]
        x, y = int((landmark.x * w - eye_x)), int((landmark.y * h - eye_y))
        if 0 <= x < eye_w and 0 <= y < eye_h: iris_points.append((x, y))
    
    iris_center_calc = fallback_iris_coords
    if len(iris_points) >= 3:
        iris_center_calc = (sum(p[0] for p in iris_points) / len(iris_points) / eye_w, 
                            sum(p[1] for p in iris_points) / len(iris_points) / eye_h)
    
    eye_region = frame[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]
    if eye_region.size == 0:
        return (iris_center_calc, iris_center_calc) if iris_center_calc is not None else (None, None)

    eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6,6)).apply(eye_gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=int(min(eye_w, eye_h) * 0.25),
                              param1=40, param2=25, minRadius=int(min(eye_w, eye_h) * 0.08), 
                              maxRadius=int(min(eye_w, eye_h) * 0.45))
    
    pupil_center = None
    if circles is not None:
        circle = np.uint16(np.around(circles))[0][0]
        pupil_center = (circle[0] / eye_w, circle[1] / eye_h)
    
    if iris_center_calc is not None:
        if pupil_center is not None:
            weight = 0.55
            final = (np.clip(weight * pupil_center[0] + (1 - weight) * iris_center_calc[0], 0, 1),
                     np.clip(weight * pupil_center[1] + (1 - weight) * iris_center_calc[1], 0, 1))
            return final, iris_center_calc
        return (np.clip(iris_center_calc[0], 0, 1), np.clip(iris_center_calc[1], 0, 1)), iris_center_calc
    
    if pupil_center is not None:
        return (np.clip(pupil_center[0], 0, 1), np.clip(pupil_center[1], 0, 1)), iris_center_calc
        
    return None, iris_center_calc

def get_eye_openness(landmarks, eye_indices):
    if len(eye_indices) < 6: return 0.5
    eye_points = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in eye_indices])
    vertical_dist = np.max(eye_points[:, 1]) - np.min(eye_points[:, 1])
    horizontal_dist = np.max(eye_points[:, 0]) - np.min(eye_points[:, 0])
    return vertical_dist / horizontal_dist if horizontal_dist > 0 else 0.15

def get_facial_axis_direction(landmarks):
    axis_points = {name: (landmarks.landmark[idx].x, landmarks.landmark[idx].y) for name, idx in FACIAL_AXIS_POINTS.items()}
    face_w = abs(axis_points['right_temple'][0] - axis_points['left_temple'][0])
    face_h = abs(axis_points['chin_tip'][1] - axis_points['forehead_center'][1])
    if face_w == 0 or face_h == 0: return 0, 0
    
    face_center_x = (axis_points['left_temple'][0] + axis_points['right_temple'][0]) / 2
    face_center_y = (axis_points['forehead_center'][1] + axis_points['chin_tip'][1]) / 2
    
    h_shift = (axis_points['nose_tip'][0] - face_center_x) / face_w * 6.5
    v_components = [
        (axis_points['nose_tip'][1] - axis_points['nose_bridge'][1]) / face_h * 12.0,
        (axis_points['nose_tip'][1] - face_center_y) / face_h * 8.5,
        ((axis_points['left_eye_center'][1] + axis_points['right_eye_center'][1])/2 - face_center_y) / face_h * 7.2,
        (axis_points['mouth_center'][1] - face_center_y) / face_h * 5.8
    ]
    return h_shift, sum(v_components)

def get_pose_direction(pose_landmarks):
    if pose_landmarks is None: return 0, 0
    try:
        nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        ls, rs = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER], pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        le, re = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR], pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
        sh_w = abs(rs.x - ls.x)
        if sh_w > 0:
            h_off = ((nose.x - (ls.x+rs.x)/2)/sh_w * 2.2) + ((nose.x - (le.x+re.x)/2)/sh_w * 2.8)
            v_tilt = (nose.y - (le.y+re.y)/2) / abs((le.y+re.y)/2 - (ls.y+rs.y)/2 + 1e-6) * 6.5
            return h_off, v_tilt
    except: pass
    return 0, 0

def combine_face_and_pose(face_h, face_v, pose_h, pose_v, pose_weight, face_axis_history):
    face_axis_history.append((face_h, face_v, pose_h, pose_v))
    avg = np.mean(list(face_axis_history), axis=0) if len(face_axis_history) > 1 else (face_h, face_v, pose_h, pose_v)
    return avg[0]*(1-pose_weight) + avg[2]*pose_weight, avg[1]*(1-pose_weight) + avg[3]*pose_weight
