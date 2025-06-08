import mediapipe as mp
import cv2
import numpy as np
from utils import read_image_from_bytes

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5
)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return 360 - angle if angle > 180 else angle

def detect_pose(landmarks):
    def get_point(name):
        lm = landmarks[mp_pose.PoseLandmark[name].value]
        return [lm.x, lm.y], lm.visibility

    def is_visible(vis, threshold=0.5):
        return vis > threshold

    # Get keypoints and visibilities
    left_shoulder, v_ls = get_point("LEFT_SHOULDER")
    right_shoulder, v_rs = get_point("RIGHT_SHOULDER")
    left_elbow, v_le = get_point("LEFT_ELBOW")
    right_elbow, v_re = get_point("RIGHT_ELBOW")
    left_wrist, v_lw = get_point("LEFT_WRIST")
    right_wrist, v_rw = get_point("RIGHT_WRIST")
    left_hip, v_lh = get_point("LEFT_HIP")
    right_hip, v_rh = get_point("RIGHT_HIP")
    left_knee, v_lk = get_point("LEFT_KNEE")
    right_knee, v_rk = get_point("RIGHT_KNEE")
    left_ankle, v_la = get_point("LEFT_ANKLE")
    right_ankle, v_ra = get_point("RIGHT_ANKLE")

    # Only predict if main points are visible
    if not all(is_visible(v) for v in [v_ls, v_rs, v_lh, v_rh, v_la, v_ra]):
        return ["Unknown"]

    # Angles
    left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    avg_hip_y = (left_hip[1] + right_hip[1]) / 2
    avg_ankle_y = (left_ankle[1] + right_ankle[1]) / 2

    vertical_span = avg_ankle_y - avg_shoulder_y
    horizontal_pose = abs(avg_shoulder_y - avg_hip_y) < 0.1

    # Standing
    if (left_leg_angle > 160 and right_leg_angle > 160 and
        left_arm_angle > 160 and right_arm_angle > 160 and
        vertical_span > 0.3 and not horizontal_pose):
        return ["Standing"]

    # Plank
    if (left_leg_angle > 160 and right_leg_angle > 160 and
        left_arm_angle > 160 and right_arm_angle > 160 and
        horizontal_pose):
        return ["Plank"]

    # Bending
    if (abs(avg_shoulder_y - avg_hip_y) < 0.05):
        return ["Bending"]

    # Squat
    if (left_leg_angle < 120 and right_leg_angle < 120):
        return ["Squat"]

    return ["Unknown"]

def get_pose_keypoints_and_annotated_image(image_bytes):
    try:
        image = read_image_from_bytes(image_bytes)
        if image is None:
            raise ValueError("Failed to read image from bytes")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            return [], "No pose detected", image

        # Draw pose landmarks on the image
        annotated_image = image.copy()
        
        # Draw landmarks with custom style (fix: do not use get_default_pose_connections_style)
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        # Add pose name to image
        detected_poses = detect_pose(results.pose_landmarks.landmark)
        pose_name = detected_poses[0] if detected_poses else "Unknown"
        
        # Add text to image
        cv2.putText(
            annotated_image,
            f"Pose: {pose_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Convert annotated image to bytes
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_image_bytes = buffer.tobytes()

        # Extract keypoints
        keypoints = [{
            "index": idx,
            "name": mp_pose.PoseLandmark(idx).name,
            "x": lm.x,
            "y": lm.y,
            "z": lm.z,
            "visibility": lm.visibility
        } for idx, lm in enumerate(results.pose_landmarks.landmark)]

        return keypoints, pose_name, annotated_image_bytes
        
    except Exception as e:
        raise Exception(f"Error in pose estimation: {str(e)}")