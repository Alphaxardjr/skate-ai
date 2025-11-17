from pose_module import poseExtractor 
from feedback_module import feedbackModule

pose = poseExtractor()
fb = feedbackModule()

video_path = "./test.mp4"

# extract angles
angles = pose.get_angles_from_video(video_path, fb_module=fb, show=True)
