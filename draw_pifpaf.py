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


general_keyareas = {}
general_keyareas["face"] = [i for i in range(5)]
general_keyareas["shoulder"] = [5,6]
general_keyareas["hip"] = [11,12]
general_keyareas["knee"] = [13,14]
general_keyareas['ankle'] = [15,16]
general_keyareas['eye'] = [2,3]

KINEMATIC_TREE_SKELETON = [
    (1, 2), (2, 4),  # left head
    (1, 3), (3, 5),
    (1, 6),
    (6, 8), (8, 10),  # left arm
    (1, 7),
    (7, 9), (9, 11),  # right arm
    (6, 12), (12, 14), (14, 16),  # left side
    (7, 13), (13, 15), (15, 17),
]
COCO_KEYPOINTS = [
    'nose',            # 1
    'left_eye',        # 2
    'right_eye',       # 3
    'left_ear',        # 4
    'right_ear',       # 5
    'left_shoulder',   # 6
    'right_shoulder',  # 7
    'left_elbow',      # 8
    'right_elbow',     # 9
    'left_wrist',      # 10
    'right_wrist',     # 11
    'left_hip',        # 12
    'right_hip',       # 13
    'left_knee',       # 14
    'right_knee',      # 15
    'left_ankle',      # 16
    'right_ankle',     # 17
]

def draw_skeleton(img, pifpaf_keypoints, predict_head):
    # openpifpaf keypoints format: (x, y, confidence)
    pp_kps = pifpaf_keypoints.reshape(-1,3)
    # draw skeleton by connecting different keypoint by coco default
    for pair in KINEMATIC_TREE_SKELETON:
        partA = pair[0] -1
        partB = pair[1] -1
        # if confidence is not zero, the keypoint exist, otherwise the keypoint would be at (0,0)
        if  not np.isclose(pp_kps[partA, 2],  0) and not np.isclose(pp_kps[partB, 2],  0):
            cv2.line(img, pp_kps[partA,:2].astype(int), pp_kps[partB,:2].astype(int), (0, 255, 255), 2)
    if predict_head:
        img = generate_head_bbox(img, pp_kps)
    return img

def get_body_bbox(pp_kps):
    min_x, min_y, max_x, max_y = 1000, 1000, 0, 0
    for i in pp_kps:
        if np.isclose(i[2], 0):
            pass
        else:
            if i[0] < min_x:
                min_x = i[0]
            if i [0] > max_x:
                max_x = i[0]
            if i[1] < min_y:
                min_x = i[1]
            if i [1] > max_y:
                max_x = i[1]
    return np.array([min_x, min_y, max_x, max_y])

def get_joint_coor(joint_name, pp_kps):
    res_x, res_y = 0, 0
    count = 0
    # ======0212 zero issue!======
    # if joint_exist(joint_name, pp_kps):
    #     pass
    for i in general_keyareas[joint_name]:
        if (np.isclose(pp_kps[i,2], 0)):
            pass
        else:
            res_x += pp_kps[i,0]
            res_y += pp_kps[i,1]
            count += 1
    if count == 0:
        return np.array([0,0])
    res_x = int(res_x/count)
    res_y = int(res_y/count)
    return np.array([res_x, res_y])

def joint_exist(joint_name, pp_kps, all_exist=False):
    general_terms = ["face", "shoulder", "hip", "knee", "ankle", "eye"]
    if joint_name in general_terms:
        num_limit = len(general_keyareas[joint_name])//2
        if all_exist:
            num_limit = 0
        if np.sum(np.isclose(pp_kps[general_keyareas[joint_name], 2], np.zeros(len(general_keyareas[joint_name])))) > num_limit:
            return False
    else:
        index_joint = COCO_KEYPOINTS.index(joint_name)
        if np.isclose(pp_kps[index_joint, 2], 0):
            return False
    return True

def get_human_height(pp_kps):
    nose2shoulder, shoulder2hip, hip2knee, knee2ankle = 0, 0, 0, 0
    if joint_exist("nose", pp_kps) and joint_exist("shoulder", pp_kps):
        nose2shoulder = np.linalg.norm(get_joint_coor("shoulder", pp_kps) - pp_kps[0,:2])
    if joint_exist("hip", pp_kps) and joint_exist("shoulder", pp_kps):
        shoulder2hip = np.linalg.norm(get_joint_coor("hip", pp_kps) - get_joint_coor("shoulder", pp_kps))
    if joint_exist("hip", pp_kps) and joint_exist("knee", pp_kps):
        hip2knee = np.linalg.norm(get_joint_coor("knee", pp_kps) - get_joint_coor("hip", pp_kps))
    if joint_exist("nose", pp_kps) and joint_exist("shoulder", pp_kps):
        knee2ankle = np.linalg.norm(get_joint_coor("ankle", pp_kps) - get_joint_coor("knee", pp_kps))
    return nose2shoulder + shoulder2hip + hip2knee + knee2ankle



def generate_head_bbox(img, pp_kps):
    head_shoulder_ratio = 1/4
    # if head exist, create a bbox
    # if shoulder exist, create a predicted bbox
    # ==== fix side people issue====
    if joint_exist("eye", pp_kps) or joint_exist("nose", pp_kps):
        max_shoulder_x, min_shoulder_x = np.amin(pp_kps[5:7, 0]).astype(int), np.amax(pp_kps[5:7, 0]).astype(int)
        head_bbox_x1, head_bbox_x2 = int((1 - head_shoulder_ratio) * min_shoulder_x + head_shoulder_ratio * max_shoulder_x), int(head_shoulder_ratio * min_shoulder_x + (1 - head_shoulder_ratio) * max_shoulder_x)
        head_bbox_y1, head_bbox_y2 = 0, 0
        if joint_exist("eye", pp_kps):
            # print("eye exist! ", pp_kps[1:3,:])
            head_bbox_y2 = int((get_joint_coor("eye", pp_kps)[1] + get_joint_coor("shoulder", pp_kps)[1])/2)
            head_bbox_y1 = int(2*get_joint_coor("eye", pp_kps)[1] - head_bbox_y2)
        else:
            # nose
            print("nose exist! ", pp_kps[0,:])
            head_bbox_y2 = int((pp_kps[0,1] + get_joint_coor("shoulder", pp_kps)[1]) /2)
            head_bbox_y1 = int(2*pp_kps[0,1] - head_bbox_y2)

        # print("xyxy: ", head_bbox_x1, head_bbox_y1, head_bbox_x2, head_bbox_y2)
        # print((head_bbox_x1, head_bbox_y1), (head_bbox_x2, head_bbox_y2) )
        cv2.rectangle(img, (head_bbox_x1, head_bbox_y1), (head_bbox_x2, head_bbox_y2), (0, 0, 255), 2)
    elif joint_exist("shoulder", pp_kps, all_exist=True):
        print("shoulder only")
        # the width of the head is about 1/3 the shoulder's width
        min_shoulder_x, max_shoulder_x = np.amin(pp_kps[5:7, 0]).astype(int), np.amax(pp_kps[5:7, 0]).astype(int)
        pred_head_bbox_x1, pred_head_bbox_x2 = int((1 - head_shoulder_ratio) * min_shoulder_x + head_shoulder_ratio * max_shoulder_x), int(head_shoulder_ratio * min_shoulder_x + (1 - head_shoulder_ratio) * max_shoulder_x)
        # human height is around 6~8 head high, take average 7
        pred_head_bbox_y1 = int(get_joint_coor("shoulder", pp_kps)[1] - get_human_height(pp_kps) / 7)
        pred_head_bbox_y2 = get_joint_coor("shoulder", pp_kps)[1]
        print("predicted xyxy: ",pred_head_bbox_x1, pred_head_bbox_y1, pred_head_bbox_x2, pred_head_bbox_y2)
        cv2.rectangle(img, (pred_head_bbox_x1, pred_head_bbox_y1), (pred_head_bbox_x2, pred_head_bbox_y2), (255, 0, 0), 2)
    return img