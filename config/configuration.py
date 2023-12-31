import os
import torch
import numpy as np


# Global device variable
#device = "cpu"  # Change this to "gpu" when needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DetectionConfig:
    # Model configuration
    model_name = "resnet50_2020-07-20"
    max_size = 2048

    # Directories
    base_dir = "artifacts"
    frame_folder = os.path.join("artifacts", "frames")
    bbox_path = os.path.join("artifacts", "bbox")
    bbox_json_path = os.path.join("artifacts", "bbox", "BBox_JSON.json")


class FrameExtraction:
    # frame rate
    fps = 6
    # Directories
    base_dir = "artifacts"
    frame_folder = os.path.join("artifacts", "frames")
    capture_mode = "video"  # or "webcam"
    video_data_path = os.path.join("artifacts", "video","Deewangi_Deewangi.mp4")


class TrackingConfig:
    # video_data_path = os.path.join("artifacts", "video","Deewangi_Deewangi.mp4")
    video_save_path = os.path.join("artifacts", "video")
    # Generate 100 random colors
    color = [(np.random.randint(0, 200), np.random.randint(0, 255), np.random.randint(0, 200)) for _ in range(100)]
    frame_resolution = (1280,720)
    fps_output = 6


#