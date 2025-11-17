class feedbackModule:
    def __init__(self):
        # Placeholder thresholds
        self.KNEE_THRESHOLD = 130
        self.TORSO_THRESHOLD = 160

    def give_feedback(self, label, angles):
        """
        label : ML classification ('pushing', 'turning', 'standing')
        angles: dict from poseExtractor
        """

        l_knee = angles["left_knee"]
        r_knee = angles["right_knee"]

        feedback = []

        # FEEDBACK RULES -------------------------

        # PUSHING
        if label == "pushing":
            if l_knee > self.KNEE_THRESHOLD:
                feedback.append("Bend your LEFT knee more.")
            if r_knee > self.KNEE_THRESHOLD:
                feedback.append("Bend your RIGHT knee more.")
            if abs(l_knee - r_knee) > 10:
                feedback.append("Try to keep your knees symmetrical.")

        # STANDING
        elif label == "standing":
            if l_knee < 160 or r_knee < 160:
                feedback.append("Straighten up while standing.")

        # TURNING 
        elif label == "turning":
            if abs(l_knee - r_knee) < 15:
                feedback.append("Lean more into the turn for stability.")

        # If NO feedback was added
        if not feedback:
            feedback.append("Good posture!")

        return feedback
