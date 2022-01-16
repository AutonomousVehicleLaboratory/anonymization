# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path
import os
import sys

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
import json
import copy


from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from tracker.sort import Sort
from config.base_config_detect_face import get_cfg_defaults

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords
def show_tracking(img, xyxyconfs, colors):
    for xyxyconf in xyxyconfs:
        x1, y1, x2, y2, conf = int(xyxyconf[0]), int(xyxyconf[1]), int(xyxyconf[2]), int(xyxyconf[3]), int(xyxyconf[4])
        cv2.rectangle(img, (x1,y1), (x2, y2), colors[conf%32], thickness=3, lineType=cv2.LINE_AA)
    return img

def show_results(img, xywh, conf, landmarks, class_num, show_landmarks = False):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    
    if show_landmarks:
        clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
        for i in range(5):
            point_x = int(landmarks[2 * i] * w)
            point_y = int(landmarks[2 * i + 1] * h)
            cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def detect_output(model, image_path, device):
    # Load model
    img_size = 800
    conf_thres = 0.3
    iou_thres = 0.5

    orgimg = cv2.imread(image_path)  # BGR
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    print('img.shape: ', img.shape)
    print('orgimg.shape: ', orgimg.shape)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)

def detect_one(model, image_path, device):
    # Load model
    img_size = 800
    conf_thres = 0.3
    iou_thres = 0.5

    orgimg = cv2.imread(image_path)  # BGR
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    print('img.shape: ', img.shape)
    print('orgimg.shape: ', orgimg.shape)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)
    if not os.path.isdir( 'output_dir' ) :
        os.mkdir( 'output_dir' )  # make sure the directory exists
    print(cv2.imwrite("./output_dir/result2.jpg", orgimg))

def detection_test(model, image_dir, device, save_dir):
    # Load model
    img_size = 800
    conf_thres = 0.3
    iou_thres = 0.5

    image_paths = sorted(os.listdir(image_dir))
    T_0 = time.time()
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    out = cv2.VideoWriter(save_dir+'output_video_cam6.avi',cv2.VideoWriter_fourcc(*'MJPG'), 3, (1920,1440))
    for image_path in image_paths:
        # print(image_path+ " start!")
        orgimg = cv2.imread(image_dir+image_path)  # BGR
        img0 = copy.deepcopy(orgimg)
        assert orgimg is not None, 'Image Not Found ' + image_path
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

        # Run inference
        t0 = time.time()

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)

        # print('img.shape: ', img.shape)
        # print('orgimg.shape: ', orgimg.shape)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
            gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

                for j in range(det.size()[0]):
                    xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                    class_num = det[j, 15].cpu().numpy()
                    orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)
        print(image_path+" done")
        # cv2.imwrite(save_dir + image_path, orgimg)
        out.write(orgimg)
        # t1 = time_synchronized()
        # print("Time: ", t1-t0)
    out.release()
    T_1 = time_synchronized()
    print("Total inference time: ", T_1 - T_0)

def detect_and_track(cfg, model, image_dir, device, save_dir, show_landmark = False, detection_path = None):
    write_video = cfg.WRITE_VIDEO
    display_results = cfg.DISPLAY_RESULTS
    
    # Load model
    img_size = cfg.IMAGE_SIZE
    conf_thres = cfg.CONF_THRES
    iou_thres = cfg.IOU_THRES

    image_paths = sorted(os.listdir(image_dir))


    T_0 = time.time()
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    detection_path = os.path.join(save_dir, 'detection.json')

    if os.path.exists(detection_path):
        print("Loading json file")
        with open(detection_path) as fp:
            detection_dict = json.load(fp)
    else:
        detection_dict = {}    
    
    if write_video:
        out = cv2.VideoWriter(save_dir+'output_tracking_cam6_both.avi',cv2.VideoWriter_fourcc(*'MJPG'), 3, (1920,1440))
    if display_results:
        cv2.namedWindow("Tracking", cv2.WND_PROP_FULLSCREEN)

    # Load Tracker
    MAX_AGE = cfg.TRACKER.MAX_AGE
    MIN_HITS = cfg.TRACKER.MIN_HITS
    IOU_THRES = cfg.TRACKER.IOU_THRES
    tracker = Sort(max_age = MAX_AGE, min_hits = MIN_HITS, iou_threshold = IOU_THRES)
    # colors = [(0,0,255)]
    colors = []
    
    for i in range(32):
        colors.append((0,0,255))
        # colors.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    
    for image_path in image_paths:
        orgimg = cv2.imread(image_dir+image_path)  # BGR
        img0 = copy.deepcopy(orgimg)
        assert orgimg is not None, 'Image Not Found ' + image_path
        
        if image_path in detection_dict:
            print('use saved detection')
            dets = []
            image_det = detection_dict[image_path]
            for det_dict in image_det:
                xywh = det_dict['xywh']
                conf = det_dict['conf']
                landmarks = det_dict['landmarks']
                class_num = det_dict['class_num']
                xyxyconf = det_dict['xyxyconf']
                dets.append(xyxyconf)
                orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)
        else:
            image_det = []
            h0, w0 = orgimg.shape[:2]  # orig hw
            r = img_size / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
                img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

            imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

            img = letterbox(img0, new_shape=imgsz)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

            # Run inference
            # t0 = time.time()

            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            
            pred = model(img)[0]

            # Apply NMS
            pred = non_max_suppression_face(pred, conf_thres, iou_thres)

            # print('img.shape: ', img.shape)
            # print('orgimg.shape: ', orgimg.shape)

            # Process detections
            dets = []
            for i, det in enumerate(pred):  # detections per image
                gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
                gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                    det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()
                    for j in range(det.size()[0]):
                        xyxyconf = det[j, :5].cpu().numpy()
                        dets.append(xyxyconf)
                        xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                        conf = det[j, 4].cpu().numpy()
                        landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                        class_num = det[j, 15].cpu().numpy()

                        det_dict = {
                            'xywh': xywh,
                            'conf': conf.tolist(),
                            'landmarks': landmarks,
                            'class_num': class_num.tolist(),
                            'xyxyconf': xyxyconf.tolist()
                        }
                        image_det.append(det_dict)
                        orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)
            detection_dict[image_path] = image_det

        # update tracker
        tracking_res = tracker.update(dets)
        orgimg = show_tracking(orgimg, tracking_res, colors)
        print(image_path+" done")

        # cv2.imwrite(save_dir + image_path, orgimg)
        if write_video:
            out.write(orgimg)
        if display_results:
            cv2.imshow("Tracking", orgimg)
            if (cv2.waitKey(0) == ord('q')):
                break
        

        # t1 = time_synchronized()
        # print("Time: ", t1-t0)
    if write_video:
        out.release()
    if not os.path.exists(detection_path):
        with open(detection_path, 'w') as fp:
            json.dump(detection_dict, fp)
    T_1 = time_synchronized()
    print("Total inference time: ", T_1 - T_0)


def detect_and_save(cfg, model, image_dir, device, save_dir, show_landmark = False, detection_path = None):
    write_video = cfg.WRITE_VIDEO
    display_results = cfg.DISPLAY_RESULTS
    
    # Load model
    img_size = cfg.IMAGE_SIZE
    conf_thres = cfg.CONF_THRES
    iou_thres = cfg.IOU_THRES

    image_paths = sorted(os.listdir(image_dir))


    T_0 = time.time()
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    detection_path = os.path.join(save_dir, 'detection.json')

    if os.path.exists(detection_path):
        print("Loading json file")
        with open(detection_path) as fp:
            detection_dict = json.load(fp)
    else:
        detection_dict = {}    
    
    if write_video:
        out = cv2.VideoWriter(save_dir+'output_tracking_cam6_both.avi',cv2.VideoWriter_fourcc(*'MJPG'), 3, (1920,1440))
    if display_results:
        cv2.namedWindow("Tracking", cv2.WND_PROP_FULLSCREEN)


    # colors = [(0,0,255)]
    colors = []
    
    for i in range(32):
        colors.append((0,0,255))
        # colors.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    
    for image_path in image_paths:
        orgimg = cv2.imread(image_dir+image_path)  # BGR
        img0 = copy.deepcopy(orgimg)
        assert orgimg is not None, 'Image Not Found ' + image_path
        
        image_det = []
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

        # Run inference
        # t0 = time.time()

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)

        # print('img.shape: ', img.shape)
        # print('orgimg.shape: ', orgimg.shape)

        # Process detections
        dets = []
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
            gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()
                for j in range(det.size()[0]):
                    xyxyconf = det[j, :5].cpu().numpy()
                    dets.append(xyxyconf)
                    xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                    class_num = det[j, 15].cpu().numpy()

                    det_dict = {
                        'xywh': xywh,
                        'conf': conf.tolist(),
                        'landmarks': landmarks,
                        'class_num': class_num.tolist(),
                        'xyxyconf': xyxyconf.tolist()
                    }
                    image_det.append(det_dict)
                    orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)
        detection_dict[image_path] = image_det

        print(image_path+" done")

        # cv2.imwrite(save_dir + image_path, orgimg)
        if write_video:
            out.write(orgimg)
        if display_results:
            cv2.imshow("Tracking", orgimg)
            if (cv2.waitKey(0) == ord('q')):
                break
        

        # t1 = time_synchronized()
        # print("Time: ", t1-t0)
    if write_video:
        out.release()
    if not os.path.exists(detection_path):
        with open(detection_path, 'w') as fp:
            json.dump(detection_dict, fp)
    T_1 = time_synchronized()
    print("Total inference time: ", T_1 - T_0)


def track_from_saved(cfg, image_dir, save_dir, show_landmark = False, detection_path = None):
    write_video = cfg.WRITE_VIDEO
    display_results = cfg.DISPLAY_RESULTS
    
    # Load model
    img_size = cfg.IMAGE_SIZE
    conf_thres = cfg.CONF_THRES
    iou_thres = cfg.IOU_THRES

    image_paths = sorted(os.listdir(image_dir))


    T_0 = time.time()
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    detection_path = os.path.join(save_dir, 'detection.json')

    if os.path.exists(detection_path):
        print("Loading detection result file")
        with open(detection_path) as fp:
            detection_dict = json.load(fp)
    else:
        detection_dict = {}    
    
    if write_video:
        out = cv2.VideoWriter(save_dir+'output_tracking_cam6_both.avi',cv2.VideoWriter_fourcc(*'MJPG'), 3, (1920,1440))
    if display_results:
        cv2.namedWindow("Tracking", cv2.WND_PROP_FULLSCREEN)

    # Load Tracker
    MAX_AGE = cfg.TRACKER.MAX_AGE
    MIN_HITS = cfg.TRACKER.MIN_HITS
    IOU_THRES = cfg.TRACKER.IOU_THRES
    tracker = Sort(max_age = MAX_AGE, min_hits = MIN_HITS, iou_threshold = IOU_THRES)
    # colors = [(0,0,255)]
    colors = []
    
    for i in range(32):
        colors.append((0,0,255))
        # colors.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    
    for image_path in image_paths:
        orgimg = cv2.imread(image_dir+image_path)  # BGR
        img0 = copy.deepcopy(orgimg)
        assert orgimg is not None, 'Image Not Found ' + image_path
        
        if image_path in detection_dict:
            print('use saved detection')
            dets = []
            image_det = detection_dict[image_path]
            for det_dict in image_det:
                xywh = det_dict['xywh']
                conf = det_dict['conf']
                landmarks = det_dict['landmarks']
                class_num = det_dict['class_num']
                xyxyconf = det_dict['xyxyconf']
                dets.append(xyxyconf)
                orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)
        else:
            print("missing prediction for image", image_path)


        # update tracker
        tracking_res = tracker.update(dets)
        orgimg = show_tracking(orgimg, tracking_res, colors)
        print(image_path+" done")

        # cv2.imwrite(save_dir + image_path, orgimg)
        if write_video:
            out.write(orgimg)
        if display_results:
            cv2.imshow("Tracking", orgimg)
            if (cv2.waitKey(0) == ord('q')):
                break
        

        # t1 = time_synchronized()
        # print("Time: ", t1-t0)
    if write_video:
        out.release()
    if not os.path.exists(detection_path):
        with open(detection_path, 'w') as fp:
            json.dump(detection_dict, fp)
    T_1 = time_synchronized()
    print("Total inference time: ", T_1 - T_0)


def parse_args():
    """ Parse the command line arguments """
    parser = argparse.ArgumentParser(description='cam_lidar_calibration')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )

    args = parser.parse_args(sys.argv[1:])
    return args


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    args = parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)


    if not cfg.TRACK_ONLY:
        if torch.cuda.is_available():
            print("=====GPU=====")
        else:
            print("=====CPU=====")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(cfg.WEIGHTS, device)
        # detect_one(model, opt.image, device)
        
        # Run detection and save the result to a json file
        # set track_only to True after running this
        detect_and_save(cfg, model, cfg.IMAGE, device, cfg.SAVE_DIR)
        
        # This will detect, track and save detection results.
        # If you run this again, the saved detection will be used
        # The code is a bit messy for this reason.
        detect_and_track(cfg, model, cfg.IMAGE, device, cfg.SAVE_DIR)
    else:
        track_from_saved(cfg, cfg.IMAGE, cfg.SAVE_DIR) # track from saved detection
