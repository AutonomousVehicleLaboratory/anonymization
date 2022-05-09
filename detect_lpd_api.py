import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
from lpd_functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


class LP_Detector():
    def __init__(self, cfg):
        self.model = self.init_model(cfg.WEIGHTS)
        self.iou_thres = cfg.IOU_THRES
        self.conf_thres = cfg.CONF_THRES
        self.size = cfg.RESIZE
    
    def init_model(self, weight):
        m = tf.saved_model.load(weight, tags=[tag_constants.SERVING])
        return m.signatures['serving_default']

    def detect(self, image, BGR=False):
        img = cv2.resize(image, (self.size, self.size))
        img /= 225.0
        pred_bbox = self.model(tf.constant(img))
        det = []
        for _, value in pred_bbox.items():
            boxes = value[:, 0:4]
            pred_conf = value[:, 4:]

            boxes, scores, _, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=self.iou_thres,
                score_threshold=self.conf_thres
            )
            for i in range(valid_detections):
                b = boxes[i]
                b.append(scores[i])
                det.append(b)
        return det

        


