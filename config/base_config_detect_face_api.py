
# The basic configuration system
import os.path as osp
import sys

# Add src directory into the path
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../")))

from yacs.config import CfgNode as CN

_C = CN()


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# Usually I will use UPPER CASE for non-parametric variables, and lower case for parametric variables because it can be
# directly pass into the function as key-value pairs.

# --------------------------------------------------------------------------- #
# General Configuration
# --------------------------------------------------------------------------- #

_C.IMAGE_DIR = '/Users/morris/Desktop/rosbags_data/2021-10-26-17-08-32/avt_cameras_camera6_image_color_compressed/'
_C.SAVE_DIR = '/Users/morris/Desktop/anonymization/output_dir/avt_cameras_camera6/'
_C.WRITE_VIDEO = False
_C.DISPLAY_RESULTS = True
_C.DISPLAY_YOLO5FACE = True
_C.DISPLAY_DEBUG_MODE = False
_C.SAVE_JSON = True

# openpifpaf config
_C.DISPLAY_PIFPAF = True
_C.PREDICT_PIFPAF_HEAD = False

# Crop the upper part of the displayed image for better visualization
_C.DISPLAY_CROP = True

# YOLOv5Face parameters

_C.FACE_DETECTOR = CN()
_C.FACE_DETECTOR.WEIGHTS = './weights/face.pt'
_C.FACE_DETECTOR.IMAGE_SIZE = 800
_C.FACE_DETECTOR.CONF_THRES = 0.3
_C.FACE_DETECTOR.IOU_THRES = 0.5

### Pose Detector parameters

_C.POSE_DETECTOR = CN()
_C.POSE_DETECTOR.WEIGHTS = 'shufflenetv2k30'


### License Plate Detector parameters

_C.LPD = CN()
_C.LPD.WEIGHTS = '/home/henry/Documents/projects/avl/anonymization/weights/lpd.pt'
_C.LPD.CONF_THRES = 0.5
_C.LPD.IOU_THRES = 0.45
_C.LPD.IMAGE_SIZE = 1024

### Tracker parameters 

_C.TRACKER = CN()
_C.TRACKER.MAX_AGE = 5
_C.TRACKER.MIN_HITS = 0
_C.TRACKER.IOU_THRES = 0.3
_C.TRACKER.DISTANCE_THRESHOLD = 100
_C.TRACKER.SIZE_DIST_RATIO = 0.3