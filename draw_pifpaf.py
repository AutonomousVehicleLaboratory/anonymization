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

from config.base_config_detect_face import get_cfg_defaults


general_keyareas = {}
general_keyareas["face"] = [i for i in range(5)]
general_keyareas["shoulder"] = [5,6]
general_keyareas["hip"] = [11,12]
general_keyareas["knee"] = [13,14]
general_keyareas['ankle'] = [15,16]
general_keyareas['eye'] = [2,3]
general_keyareas['torso'] = [5,6,11,12]
general_keyareas['right_body'] = [2, 4,6,8,10,12,14,16]
general_keyareas['left_body'] = [1,3,5,7,9,11,13,15]

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
    'nose',            # 0
    'left_eye',        # 1
    'right_eye',       # 2
    'left_ear',        # 3
    'right_ear',       # 4
    'left_shoulder',   # 5
    'right_shoulder',  # 6
    'left_elbow',      # 7
    'right_elbow',     # 8
    'left_wrist',      # 9
    'right_wrist',     # 10
    'left_hip',        # 11
    'right_hip',       # 12
    'left_knee',       # 13
    'right_knee',      # 14
    'left_ankle',      # 15
    'right_ankle',     # 16
]

def draw_skeleton(img, pifpaf_keypoints, predict_head, pifpaf_bbox):
    # openpifpaf keypoints format: (x, y, confidence)
    pp_kps = pifpaf_keypoints.reshape(-1,3)
    # draw skeleton by connecting different keypoint by coco default
    for pair in KINEMATIC_TREE_SKELETON:
        partA = pair[0] -1
        partB = pair[1] -1
        # left
        color = (0, 255, 255)
        # right
        if partA % 2 ==0 and partB%2==0:
            color = (255,0,255)
        # if confidence is not zero, the keypoint exist, otherwise the keypoint would be at (0,0)
        if  not np.isclose(pp_kps[partA, 2],  0) and not np.isclose(pp_kps[partB, 2],  0):
            cv2.line(img, pp_kps[partA,:2].astype(int), pp_kps[partB,:2].astype(int), color, 2)
    if predict_head:
        box, box_from_face, conf = generate_head_bbox(pp_kps, pifpaf_bbox)
        if box is not None:
            color = (0, 0, 255) if box_from_face else (255, 255, 0)
            cv2.rectangle(img, box[0], box[1], color, 2)
    return img

def face_to_us(pp_kps):
    left_x = get_joint_coor("left_body", pp_kps)[0]
    right_x = get_joint_coor("right_body", pp_kps)[0]
    if left_x < right_x:
        return False
    return True

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
    res_x, res_y, conf = 0, 0, 0
    count = 0
    for i in general_keyareas[joint_name]:
        if (np.isclose(pp_kps[i,2], 0)):
            pass
        else:
            res_x += pp_kps[i,0]
            res_y += pp_kps[i,1]
            conf += pp_kps[i,2]
            count += 1
    if count == 0:
        return np.array([0,0])
    res_x = int(res_x/count)
    res_y = int(res_y/count)
    conf /= count
    return np.array([res_x, res_y, conf])

def joint_exist(joint_name, pp_kps, all_exist=False):
    # general_terms = ["face", "shoulder", "hip", "knee", "ankle", "eye"]
    if joint_name in general_keyareas.keys():
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
# change to voting method
def get_human_height(pp_kps):
    n_s_to_h_ratio, torso_to_h_ratio, hip_knee_to_h_ratio, knee_ankle_to_h_ratio = 0.12, 0.3,  0.25, 0.25
    predicted_height = []
    height = 0
    if joint_exist("nose", pp_kps) and joint_exist("shoulder", pp_kps):
        nose2shoulder = np.linalg.norm(get_joint_coor("shoulder", pp_kps) - pp_kps[0,:2])
        predicted_height.append(nose2shoulder/n_s_to_h_ratio)
    if joint_exist("hip", pp_kps) and joint_exist("shoulder", pp_kps):
        shoulder2hip = np.linalg.norm(get_joint_coor("hip", pp_kps) - get_joint_coor("shoulder", pp_kps))
        predicted_height.append(shoulder2hip / torso_to_h_ratio)
    if joint_exist("hip", pp_kps) and joint_exist("knee", pp_kps):
        hip2knee = np.linalg.norm(get_joint_coor("knee", pp_kps) - get_joint_coor("hip", pp_kps))
        predicted_height.append(hip2knee / hip_knee_to_h_ratio)
    if joint_exist("ankle", pp_kps) and joint_exist("knee", pp_kps):
        knee2ankle = np.linalg.norm(get_joint_coor("ankle", pp_kps) - get_joint_coor("knee", pp_kps))
        predicted_height.append(knee2ankle / knee_ankle_to_h_ratio)
    # print(predicted_height)
    if predicted_height:
        height = np.median(np.array(predicted_height))
    return height


def generate_head_bbox(pp_kps, pp_bboxes):
    torso_length_head_width_ratio = 2/5
    neck_to_head_height_ratio = 1/4
    head_aspect_ratio = 1.2
    if face_to_us(pp_kps):
        if joint_exist("face", pp_kps)  and joint_exist("shoulder", pp_kps):
            head_width = 0
            # max_shoulder_x, min_shoulder_x = np.amin(pp_kps[5:7, 0]).astype(int), np.amax(pp_kps[5:7, 0]).astype(int)
            # head_bbox_x1, head_bbox_x2 = int((1 - head_shoulder_ratio) * min_shoulder_x + head_shoulder_ratio * max_shoulder_x), int(head_shoulder_ratio * min_shoulder_x + (1 - head_shoulder_ratio) * max_shoulder_x)
            head_middle_coor = get_joint_coor('face', pp_kps)[:2]
            conf = 0
            if joint_exist("hip", pp_kps):
                head_width = (get_joint_coor("hip",pp_kps)[1] - get_joint_coor("shoulder",pp_kps)[1])*torso_length_head_width_ratio
                conf = (get_joint_coor("hip",pp_kps)[2] + get_joint_coor("shoulder",pp_kps)[2]) /2
            elif joint_exist("shoulder", pp_kps, all_exist=True):
                head_width = (np.amax(pp_kps[5:7, 0]) - np.amin(pp_kps[5:7, 0])) /1.5
                conf = np.mean(pp_kps[5:7,2])
            else:
                head_width = get_joint_coor("shoulder", pp_kps)[1] - get_joint_coor("face", pp_kps)[1]
                conf = (get_joint_coor("shoulder", pp_kps)[2] + get_joint_coor("face", pp_kps)[2])/2
            head_bbox_x1, head_bbox_x2 = int(head_middle_coor[0] - head_width/2), int(head_middle_coor[0] + head_width/2)
            head_bbox_y1, head_bbox_y2 = int(head_middle_coor[1] - head_width * head_aspect_ratio/2), int(head_middle_coor[1] + head_width * head_aspect_ratio/2)
            # print("xyxy: ", head_bbox_x1, head_bbox_y1, head_bbox_x2, head_bbox_y2)
            # print((head_bbox_x1, head_bbox_y1), (head_bbox_x2, head_bbox_y2) )
            box = ((head_bbox_x1, head_bbox_y1), (head_bbox_x2, head_bbox_y2))
            box_from_face = True
        elif joint_exist("shoulder", pp_kps, all_exist=True):
            # 
            head_middle_x, head_width  = get_joint_coor("shoulder", pp_kps)[0], (np.amax(pp_kps[5:7, 0]) - np.amin(pp_kps[5:7, 0])) /1.5
            head_height = head_width * head_aspect_ratio
            conf = get_joint_coor("shoulder", pp_kps)[2]
            if joint_exist("hip", pp_kps):
                head_width = (get_joint_coor("hip",pp_kps)[1] - get_joint_coor("shoulder",pp_kps)[1])*torso_length_head_width_ratio
                conf = (get_joint_coor("hip",pp_kps)[2] + get_joint_coor("shoulder",pp_kps)[2]) /2
            pred_head_bbox_x1, pred_head_bbox_x2 = int(head_middle_x - head_width/2), int(head_middle_x + head_width/2)
            # human height is around 6~8 head high, take average 7
            # pred_head_bbox_y1 = int(get_joint_coor("shoulder", pp_kps)[1] - (pred_head_bbox_x2 - pred_head_bbox_x1))
            pred_human_height = get_human_height(pp_kps)
            if pred_human_height!=0:
                head_height = pred_human_height / 5.5
            pred_head_bbox_y2 = int(get_joint_coor("shoulder", pp_kps)[1] +- head_height * neck_to_head_height_ratio) 
            pred_head_bbox_y1 = int(pred_head_bbox_y2 - head_height)
            # print("predicted xyxy: ",pred_head_bbox_x1, pred_head_bbox_y1, pred_head_bbox_x2, pred_head_bbox_y2)
            box = ((pred_head_bbox_x1, pred_head_bbox_y1), (pred_head_bbox_x2, pred_head_bbox_y2))
            box_from_face = False
        
        else:
            box = None
            box_from_face = None
            conf = None
    else:
        box = None
        box_from_face = None
        conf = None
        # else, get body left most and right most
    return box, box_from_face, conf

def predict_and_save(save_dir):
    pifpaf_path = os.path.join(save_dir, 'detection_pifpaf.json')
    torso_length_head_width_ratio = 1/2
    head_aspect_ratio = 1.2
    detection_dict = {}
    with open(pifpaf_path) as fpif:
        pifpaf_dict = json.load(fpif)
    for img_dest, pp_dicts in pifpaf_dict.items():
        image_pred = []
        for pp_dict in pp_dicts:
            pp_kps = np.asarray(pp_dict['keypoints'])
            pp_kps = pp_kps.reshape(-1,3)
            box, box_from_face, conf = generate_head_bbox(pp_kps)
            if box is not None:
                det_dict = {"xyxyconf": [box[0][0], box[0][1], box[1][0], box[1][1], round(conf, 4)], "box_from_face": box_from_face}
                image_pred.append(det_dict)
        detection_dict[img_dest] = image_pred
    output_path = os.path.join(save_dir, "pifpaf_pred_head_0310.json")
    if os.path.exists(output_path):
        print("A file with the same name as output file exists")
    else:
        with open(output_path, 'w') as f:
            json.dump(detection_dict, f)
        print("head prediction saved!")


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

    save_dir = cfg.SAVE_DIR

    predict_and_save(save_dir)