import cv2
import mediapipe as mp
import numpy as np
import time

class poseExtractor:
    def __init__(self, detect_confidence=0.5, track_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=detect_confidence,
            min_tracking_confidence=track_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def find_position(self, img):
        landmarks = []
        if self.results and self.results.pose_landmarks:
            h, w, _ = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((id, cx, cy))
        return landmarks

    def calculate_angle(self, a, b, c):
        """Compute the angle between three points (in pixels)."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def get_angles_from_video(self, video_path, fb_module=None, show=True):
        """Read video, calculate joint angles, overlay angles and feedback."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open {video_path}")
            return []

        angles_per_frame = []

        while True:
            success, frame = cap.read()
            if not success:
                break

            self.find_pose(frame, draw=False)
            landmarks = self.find_position(frame)

            feedback_text = []

            if len(landmarks) > 28:  
                # Left knee
                lh = landmarks[23][1:]
                lk = landmarks[25][1:]
                la = landmarks[27][1:]
                left_knee_angle = self.calculate_angle(lh, lk, la)

                # Right knee
                rh = landmarks[24][1:]
                rk = landmarks[26][1:]
                ra = landmarks[28][1:]
                right_knee_angle = self.calculate_angle(rh, rk, ra)

                angles_per_frame.append({"left_knee": left_knee_angle, "right_knee": right_knee_angle})

                # Draw lines on legs
                if show:
                    cv2.line(frame, lh, lk, (255, 255, 255), 2)
                    cv2.line(frame, lk, la, (255, 255, 255), 2)
                    cv2.line(frame, rh, rk, (255, 255, 255), 2)
                    cv2.line(frame, rk, ra, (255, 255, 255), 2)

                    # Overlay angles
                    cv2.putText(frame, f"L Knee: {int(left_knee_angle)}°", (50, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"R Knee: {int(right_knee_angle)}°", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Generate feedback 
                if fb_module:
                   
                    label = "pushing"  
                    feedback_text = fb_module.give_feedback(label, {"left_knee": left_knee_angle,
                                                                    "right_knee": right_knee_angle})

                    if show:
                        y0 = 140
                        for i, text in enumerate(feedback_text):
                            y = y0 + i*30
                            cv2.putText(frame, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                if show:
                    cv2.putText(frame, "No pose detected", (50, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if show:
                cv2.namedWindow("skate-ai", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("skate-ai", 800, 600)
                cv2.imshow("skate - ai", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return angles_per_frame
