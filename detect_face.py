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
from utils.draw_pifpaf import draw_skeleton, predict_and_draw_head


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
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    for xyxyconf in xyxyconfs:
        x1, y1, x2, y2 = int(xyxyconf[0]), int(xyxyconf[1]), int(xyxyconf[2]), int(xyxyconf[3])
        trk_id, hit_streak, hits, conf = int(xyxyconf[4]), int(xyxyconf[6]), int(xyxyconf[7]), float(xyxyconf[8])
        
        cv2.rectangle(img, (x1,y1), (x2, y2), colors[trk_id%32], thickness=3, lineType=cv2.LINE_AA)
        track_str = "ID" + str(trk_id) + " HS" + str(hit_streak) + " H" + str(hits)
        cv2.putText(img, track_str, (x1, y2 + int(tl) + 12), 0, tl/2, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    return img


def show_results(img, xywh, conf=None, landmarks=None, class_num = 1, show_landmarks = False):
    h,w,c = img.shape
    tl = 2 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    
    if show_landmarks and landmarks != None:
        clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
        for i in range(5):
            point_x = int(landmarks[2 * i] * w)
            point_y = int(landmarks[2 * i + 1] * h)
            cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    # cv2.putText(img, label, (x1, y1 - 2), 0, tl / 2, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def show_results_xyxys(img, xyxys, mode="roi"):
    mode2color = {"roi":(0,255,0), "face":(255,0,0), "head":(0,0,255)}
    color = mode2color[mode]
    for xyxy in xyxys:
        h,w,_ = img.shape
        tl = 2 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
        x1 = int(xyxy[0])
        y1 = int(xyxy[1])
        x2 = int(xyxy[2])
        y2 = int(xyxy[3])
        cv2.rectangle(img, (x1,y1), (x2, y2), color=color, thickness=tl, lineType=cv2.LINE_AA)
    

    return img

def show_results_xyxy(img, xyxy):
    h,w,c = img.shape
    tl = 2 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    return img


def view_tracking_with_history(img_list, display_crop = False):
    img_id = len(img_list) -1
    while img_id >= 0 and img_id < len(img_list):
        img = img_list[img_id]
        if display_crop:
            # img_cropped = img[int(img.shape[0]*2/5)::]
            img_cropped = img#[625:840, 475:740]
        else:
            img_cropped = img
        # command_text = "q:Quit, p:Previous, n:Next, others:continue tracking"
        # cv2.putText(img_cropped, command_text, (30,30), 0, 0.9, [0,255,255], 2, cv2.LINE_AA)
        cv2.imshow("Tracking", img_cropped)
        if (cv2.waitKey(0) == ord('q')):
            exit(0)
        elif (cv2.waitKey(0) == ord('p')):
            if img_id > 0:
                img_id = img_id - 1
        elif (cv2.waitKey(0) == ord('n')):
            if img_id + 1 < len(img_list):
                img_id = img_id + 1
        else:
            break

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
        out = cv2.VideoWriter(save_dir+'output_tracking_cam6_pifpaf_face.avi',cv2.VideoWriter_fourcc(*'MJPG'), 3, (1920,1440))
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


def format_tracking_result(tracking_res):
    tracker_list = []
    for track in tracking_res:
        track_dict = {}
        track_dict["id"] = int(track[4])
        track_dict["xyxyconf"] = [
            int(track[0]), int(track[1]), int(track[2]), int(track[3]),
            float(track[8])
        ]
        tracker_list.append(track_dict)
    return tracker_list


def fuse_by_confidence(labels_yolo, labels_pifpaf):
    box_filtered = []
    for box_yolo in labels_yolo:
        in_a_box_and_low_conf = False
        for box_pifpaf in labels_pifpaf:
            if box_yolo[0] >= box_pifpaf[0] and \
            box_yolo[2] <= box_pifpaf[2] and \
            box_yolo[1] >= box_pifpaf[1] and \
            box_yolo[3] <= box_pifpaf[3]:
                if box_yolo[-1] < box_pifpaf[-1]:
                    in_a_box_and_low_conf = True
                    break
        if not in_a_box_and_low_conf:
            box_filtered.append(box_yolo)
    for box_pifpaf in labels_pifpaf:
        has_a_box_and_low_conf = False
        for box_yolo in labels_yolo:
            if box_yolo[0] >= box_pifpaf[0] and \
            box_yolo[2] <= box_pifpaf[2] and \
            box_yolo[1] >= box_pifpaf[1] and \
            box_yolo[3] <= box_pifpaf[3]:
                if box_yolo[-1] > box_pifpaf[-1]:
                    has_a_box_and_low_conf = True
                    break
        if not has_a_box_and_low_conf:
            box_filtered.append(box_pifpaf)
    return box_filtered

def filter_boxes(labels_yolo, labels_pifpaf, method):
    box_filtered = []
    if method == 'remove_face':
        for label_yolo in labels_yolo:
            box_yolo = label_yolo['xyxyconf']
            in_a_box = False
            for label_pifpaf in labels_pifpaf:
                box_pifpaf = label_pifpaf['xyxyconf']
                if box_yolo[0] >= box_pifpaf[0] and \
                box_yolo[2] <= box_pifpaf[2] and \
                box_yolo[1] >= box_pifpaf[1] and \
                box_yolo[3] <= box_pifpaf[3]:
                    in_a_box = True
                    break
            if not in_a_box:
                box_filtered.append(label_yolo)
    elif method == 'remove_head':
        for label_pifpaf in labels_pifpaf:
            box_pifpaf = label_pifpaf['xyxyconf']
            has_a_box = False
            for label_yolo in labels_yolo:
                box_yolo = label_yolo['xyxyconf']
                if box_yolo[0] >= box_pifpaf[0] and \
                box_yolo[2] <= box_pifpaf[2] and \
                box_yolo[1] >= box_pifpaf[1] and \
                box_yolo[3] <= box_pifpaf[3]:
                    has_a_box = True
                    break
            if not has_a_box:
                box_filtered.append(label_pifpaf)
    elif method == 'conf_fusion':
        box_filtered = fuse_by_confidence(labels_yolo, labels_pifpaf)
    else:
        box_filtered.extend(labels_pifpaf)

    return box_filtered



def merge_detection(det_yolo, det_pifpaf, method='remove_face'):
    detection_merged = []
    if method == 'remove_face':
        detection_merged.extend(det_pifpaf)
        if len(det_yolo) > 0:
            filtered_boxes = filter_boxes( det_yolo, det_pifpaf, method)
            detection_merged.extend(filtered_boxes)
    elif method == 'conf_fusion':
        detection_merged = filter_boxes( det_yolo, det_pifpaf, method)
    else:
        detection_merged.extend(det_yolo)
        if len(det_pifpaf) > 0:
            filtered_boxes = filter_boxes( det_yolo, det_pifpaf, method)
            detection_merged.extend(filtered_boxes) 
    return detection_merged


def anonymize_detections(img, detections):
    
    limit = img.shape 
    for box in detections:
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

def track_from_saved(cfg, image_dir, save_dir, show_landmark = False, detection_path = None):
    write_video = cfg.WRITE_VIDEO
    display_results = cfg.DISPLAY_RESULTS
    save_tracking = cfg.SAVE_TRACKING

    image_paths = sorted(os.listdir(image_dir))

    T_0 = time.time()
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    detection_path = os.path.join(save_dir, 'detection.json')
    pifpaf_path = os.path.join(save_dir, 'detection_pifpaf.json')
    pred_head_path = os.path.join(save_dir, 'pifpaf_pred_head.json')
    
    if os.path.exists(detection_path):
        print("Loading detection result file")
        with open(detection_path) as fp:
            detection_dict = json.load(fp)
        with open(pifpaf_path) as fpif:
            pifpaf_dict = json.load(fpif)
        with open(pred_head_path) as fpred:
            pred_head_dict = json.load(fpred)
    else:
        detection_dict = {}
        pifpaf_dict = {}
    
    if write_video:
        out = cv2.VideoWriter(save_dir+'output_tracking_conf_fused_blurred_testing.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10, (1920,1440))
    if display_results:
        cv2.namedWindow("Tracking", cv2.WND_PROP_FULLSCREEN)
    if save_tracking:
        tracking_result_dict = {}

    # Load Tracker
    tracker = Sort(max_age = cfg.TRACKER.MAX_AGE, 
                   min_hits = cfg.TRACKER.MIN_HITS, 
                   iou_threshold = cfg.TRACKER.IOU_THRES,
                   distance_threshold = cfg.TRACKER.DISTANCE_THRESHOLD,
                   ratio = cfg.TRACKER.SIZE_DIST_RATIO)
    colors = []
    
    for i in range(32):
        colors.append((0,0,255))

    img_list = []
    for image_path in image_paths:
        if image_path not in detection_dict:
            continue
        if write_video or display_results:
            orgimg = cv2.imread(os.path.join(image_dir, image_path))  # BGR
            assert orgimg is not None, 'Image Not Found ' + image_path
        
        if image_path in detection_dict:
            print('use saved detection')
            dets = []
            image_det = detection_dict[image_path]
            pifpaf_det = pifpaf_dict[image_path]
            pif_paf_pred = pred_head_dict[image_path]
            det_pifpaf_head, det_yolo5face = [], []
            
            if (cfg.DISPLAY_PIFPAF):
                for pp_dict in pifpaf_det:
                    pp_kps = np.asarray(pp_dict['keypoints'])
                    # if display_results or write_video:
                    #     orgimg = draw_skeleton(orgimg, pp_kps))
                    # if cfg.PREDICT_PIFPAF_HEAD:
                    #     detect_and_draw_head(orgimg, pp_kps)
                for pred_dict in pif_paf_pred:
                    xyxyconf = pred_dict['xyxyconf']
                    dets.append(xyxyconf)
                    det_pifpaf_head.append(xyxyconf)
            if (cfg.DISPLAY_YOLO5FACE):
                for det_dict in image_det:
                    xywh = det_dict['xywh']
                    conf = det_dict['conf']
                    landmarks = det_dict['landmarks']
                    class_num = det_dict['class_num']
                    xyxyconf = det_dict['xyxyconf']
                    dets.append(xyxyconf)
                    det_yolo5face.append(xyxyconf)
                    # if display_results or write_video:
                    #     orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)
            if (cfg.DISPLAY_PIFPAF and cfg.DISPLAY_YOLO5FACE):
                detection_merged = merge_detection(det_yolo5face, det_pifpaf_head, method='conf_fusion')
                if display_results or write_video:
                    anonymize_detections(orgimg, detection_merged)
                    for det_merged in detection_merged:
                        orgimg = show_results_xyxy(orgimg, det_merged)
        else:
            print("missing prediction for image", image_path)
            exit(0)

        # update tracker
        tracking_res = tracker.update(dets)
        # orgimg = show_tracking(orgimg, tracking_res, colors)
        print(image_path+" done")

        if write_video:
            out.write(orgimg)
        if display_results:# and image_path=='1635293319.560385704.jpg':
            img_list.append(orgimg)
            view_tracking_with_history(img_list, display_crop=cfg.DISPLAY_CROP)
        if save_tracking:
            tracking_result_dict[image_path] = format_tracking_result(tracking_res)


        # t1 = time_synchronized()
        # print("Time: ", t1-t0)
    if write_video:
        out.release()
    if save_tracking:
        tracking_path = os.path.join(save_dir, 'tracking_result.json')
        if not os.path.exists(tracking_path):
            with open(tracking_path, 'w') as fp:
                json.dump(tracking_result_dict, fp)
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
