# The basic configuration system
import os.path as osp
import sys

# Add src directory into the path
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../")))

from yacs.config import CfgNode as CN

_C = CN()


def get_cfg_default():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

_C.IMAGE_DIR = '/Users/morris/Desktop/rosbags_data/2021-10-26-17-08-32/avt_cameras_camera6_image_color_compressed/'

_C.SAVE_DIR = '/Users/morris/Desktop/anonymization/output_dir/avt_cameras_camera6/'

# _C.TRACK_ONLY = True

# _C.WRITE_VIDEO = False

# _C.DISPLAY_RESULTS = True
# _C.DISPLAY_YOLO5FACE = True
# # openpifpaf config
# _C.DISPLAY_PIFPAF = True
# _C.PREDICT_PIFPAF_HEAD = False

# Crop the upper part of the displayed image for better visualization
_C.DISPLAY_CROP = True

_C.LPD = CN()
_C.LPD.WEIGHTS= '/home/jliao/anonymization/weights/lpd.pt'
_C.LPD.CONF_THRES= 0.5
_C.LPD.IOU_THRES= 0.45
_C.LPD.IMAGE_SIZE= 928