"""Reprocess the intermediate results to produce better results.
This is useful because we can reprocess without running the expensive inference step.
Given that YOLO5Face and OpenPifPaf results are saved into the json file, we can adjust
the inferred head or fusion methods easily."""


import os
import sys
import json
import argparse
import glob
import cv2
import numpy as np

from utils.draw_pifpaf import draw_skeleton


def parse_args():
    """ Parse the command line arguments """
    parser = argparse.ArgumentParser(description='visualize detection results')
    parser.add_argument(
        "--dir", 
        type=str, 
        help='the directory of extracted rosbag data')

    args = parser.parse_args(sys.argv[1:])
    return args


def fuse_detections(det_yolo, det_pifpaf, method='conf_fusion'):
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


def generate_new_head(head_boxes, shrink_head_ratio=0.8):
    new_head_boxes = []
    ratio = shrink_head_ratio / 2 + 0.5
    for head in head_boxes:
        new_head = [
            ratio * head[0] + (1-ratio) * head[2],
            ratio * head[1] + (1-ratio) * head[3],
            (1-ratio) * head[0] + ratio * head[2],
            (1-ratio) * head[1] + ratio * head[3],
            head[4]
        ]
        new_head_boxes.append(new_head)
    return np.array(new_head_boxes)



def rewrite_json(det_dict, fusion_method='conf_fusion'):
    det_dict_new = {}
    for image_name in det_dict:
        # print(det_frame_dict['lp'])
        det_frame_dict = det_dict[image_name]
        
        roi = det_frame_dict['roi']
        face = np.array(det_frame_dict['face'])
        head = np.array(det_frame_dict['head'])
        pose = det_frame_dict['pose']
        lp = det_frame_dict['lp']

        head_new = generate_new_head(head, shrink_head_ratio=0.8)
        roi_new = fuse_detections(face, head_new, method=fusion_method)

        det_frame_dict_new = {}
        det_frame_dict_new['roi'] = [item.tolist() for item in roi_new]
        det_frame_dict_new['face'] = [item.tolist() for item in face]
        det_frame_dict_new['head'] = [item.tolist() for item in head_new]
        det_frame_dict_new['pose'] = pose
        det_frame_dict_new['lp'] = lp
        det_dict_new[image_name] = det_frame_dict_new

    return det_dict_new


def test_rewrite_json():
    args = parse_args()

    print("dir: ", args.dir)
    pattern = os.path.join(args.dir, '**/*_det.json') 
    print("pattern: ", pattern)

    for file_path in glob.glob(pattern, recursive=True):
        # print(file_path)
        # continue
        with open(file_path, 'r') as fp:
            det_dict = json.load(fp)
            print('detection json file loaded from ', file_path)
            det_dict_new = rewrite_json(det_dict)
            file_new_path = file_path[0:-5] + '_new.json'
            with open(file_new_path, 'w') as fp:
                json.dump(det_dict_new, fp)
                print("new detection json written to", file_new_path)


if __name__ == '__main__':
    test_rewrite_json()
