### Knee Angle detection and calculation

``` python 

    landmarks = []
    h, w, _ = img.shape
    for id, lm in enumerate(results.pose_landmarks.landmark):
        cx, cy = int(lm.x * w), int(lm.y * h)
        landmarks.append((id, cx, cy))


        if len(landmarks) > 28:  # Pose detected
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


```