import argparse
import time
from pathlib import Path

import copy
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from config.base_config_lpd_api import get_cfg_default

class LP_Anonymizer():
    def __init__(self, cfg):
        cfg = self.get_default_cfg(cfg)
        self.model, self.device, self.stride = self.init_model(cfg.LPD.WEIGHTS)
        self.iou_thres = cfg.LPD.IOU_THRES
        self.conf_thres = cfg.LPD.CONF_THRES
        self.size = cfg.LPD.IMAGE_SIZE

    def init_model(self, weights):
        if torch.cuda.is_available():
            print("=====GPU=====")
        else:
            print("=====CPU=====")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max()) 
        return model, device, stride

    def get_default_cfg(self, cfg):
        if cfg is None:
            cfg = get_cfg_default()
        elif type(cfg) is str:
            cfg_file = cfg
            cfg = get_cfg_default()
            cfg.merge_from_file(cfg_file)
        return cfg

    def detect(self, image, BGR=False):
        # print("start!")
        img0 = copy.deepcopy(image)
        imgsz = check_img_size(self.size, s=self.stride)  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]

        # # Convert
        if BGR:
            img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

        # Run inference
        # t0 = time.time()

        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        dets=[]
        for i, det in enumerate(pred):
            gn = torch.tensor(image.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                for j in range(det.size()[0]):
                    #class 0 is lp
                    if np.isclose(det[j, 5].cpu().numpy(), 0):
                        xyxyconf = det[j, :5].cpu().numpy()
                        dets.append(xyxyconf)
        # print("end!")
        return dets
    def anonymize_rois(self, img, rois):
        limit = img.shape 
        for box in rois:
            kx = int(max(box[2] - box[0], box[3] - box[1]) / 8) *2 + 1
            kx = max(5, kx)
            ksize = (kx, kx)
            sigmaX = int(kx / 2)
            color = (0,255,0)
            box[0] = 0 if box[0] < 0 else box[0]
            box[1] = 0 if box[1] < 0 else box[1]
            box[2] = limit[1]-1 if box[2] > limit[1] else box[2]
            box[3] = limit[0]-1 if box[3] > limit[0] else box[3]
            img[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = \
                cv2.GaussianBlur(
                    img[int(box[1]):int(box[3]), int(box[0]):int(box[2])],
                    ksize,
                    sigmaX)

        


if __name__ =='__main__':
    # w = torch.load("/home/jliao/Vehicle-number-plate-recognition-YOLOv5/yolov5/runs/train/exp6/weights/best.pt")
    # # print(type(w['model']))
    # for k in w: print(k, type(w[k]))
    # for k in w['shared_layers']: print("Shared layer", k)
    anonymizer_path = '/home/jliao/anonymization'
    LPD_config_file = anonymizer_path + '/config/lpd.yaml'
    lpd_anonymizer = LP_Anonymizer(LPD_config_file)
    img_path = "/home/jliao/rosbags/2020-08-17-19-02-50.bag/camera6/2481.png"
    img = cv2.imread(img_path)
    res = lpd_anonymizer.detect(img, True)
    print(res)