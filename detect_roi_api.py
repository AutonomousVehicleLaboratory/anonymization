from concurrent.futures import process
import sys
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import glob
import re
import json
from anonymization.detect_face import show_results, show_results_xyxys

from config.base_config_detect_face_api import get_cfg_defaults
from detect_face_api import Face_Detector
from detect_pose_api import Pose_Detector
from detect_lp_api import LP_Detector
from draw_pifpaf import generate_head_bbox, draw_skeleton


class Face_Anonymizer():

    def __init__(self, cfg=None):
        cfg = self.get_default_cfg(cfg)
        self.face_detector = Face_Detector(cfg.FACE_DETECTOR)
        self.pose_detector = Pose_Detector(cfg.POSE_DETECTOR)
        self.lp_detector = LP_Detector(cfg.LPD)
        self.fusion_method = 'conf_fusion'
        self.save_dir = Path(self.increment_path(Path(cfg.SAVE_DIR) / "exp", exist_ok=False))
        self.save_json = cfg.SAVE_TRACKING
        self.viz = cfg.DISPLAY_RESULTS
        self.show_pifpaf = cfg.DISPLAY_PIFPAF
        self.debug_mode_show = cfg.DISPLAY_DEBUG_MODE


    def get_default_cfg(self, cfg):
        if cfg is None:
            cfg = get_cfg_defaults()
        elif type(cfg) is str:
            cfg_file = cfg
            cfg = get_cfg_defaults()
            cfg.merge_from_file(cfg_file)
        return cfg

    def increment_path(path, exist_ok=True, sep=''):
        # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
        path = Path(path)  # os-agnostic
        if (path.exists() and exist_ok) or (not path.exists()):
            return str(path)
        else:
            dirs = glob.glob(f"{path}{sep}*")  # similar paths
            matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]  # indices
            n = max(i) + 1 if i else 2  # increment number
            return f"{path}{sep}{n}"  # update path

    def detect_roi(self, image, BGR=False):
        """ Given an image, return the bounding boxes """
        # from timeit import default_timer as timer
        # start1 = timer()
        dets_face = self.face_detector.detect(image, BGR=BGR)
        # end1 = timer()
        # start2 = timer()
        dets_pose = self.pose_detector.detect(image)
        # end2 = timer()
        dets_lp = self.lp_detector.detect(image, BGR=BGR)
        # print('time:', end1 - start1, end2 - start2)
        dets_head = self.generate_head_from_pose(dets_pose)
        dets_roi = self.fuse_detections(dets_face, dets_head, method=self.fusion_method)
        if self.save_json:
            with open(os.path.join(self.save_dir, "detection_pifpaf.json"),'w') as fp:
                json.dump(dets_pose)
            with open(os.path.join(self.save_dir, "detection_face.json"),'w') as ff:
                json.dump(dets_face)
            with open(os.path.join(self.save_dir, "detection_lp.json"),'w') as flp:
                json.dump(dets_pose)
            with open(os.path.join(self.save_dir, "detection_head_fusion.json"),'w') as froi:
                json.dump(dets_roi)
        if self.viz:
            show_results_xyxys(image, dets_roi)
        if self.debug_mode_show:
            show_results_xyxys(image, dets_face, mode="face")
            show_results_xyxys(image, dets_head, mode="head")
        if self.show_pifpaf:
            for pp_dict in dets_pose:
                pp_kps = np.asarray(pp_dict['keypoints'])
                draw_skeleton(image, pp_kps)
        return dets_roi, dets_lp


    def generate_head_from_pose(self, dets_pose):
        dets_head = []
        for pose in dets_pose:
            pp_kps = pose["keypoints"].reshape(-1,3)
            box, box_from_face, conf = generate_head_bbox(pp_kps)
            if box is not None:
                dets_head.append(
                    np.array([box[0][0], box[0][1], box[1][0], box[1][1], conf])
                )
        return dets_head

    
    def fuse_detections(self, det_yolo, det_pifpaf, method='conf_fusion'):
        detection_merged = []
        if method == 'remove_face':
            detection_merged.extend(det_pifpaf)
            if len(det_yolo) > 0:
                filtered_boxes = self.filter_boxes( det_yolo, det_pifpaf, method)
                detection_merged.extend(filtered_boxes)
        elif method == 'conf_fusion':
            detection_merged = self.filter_boxes( det_yolo, det_pifpaf, method)
        else:
            detection_merged.extend(det_yolo)
            if len(det_pifpaf) > 0:
                filtered_boxes = self.filter_boxes( det_yolo, det_pifpaf, method)
                detection_merged.extend(filtered_boxes) 
        return detection_merged


    def fuse_by_confidence(self, labels_yolo, labels_pifpaf):
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


    def filter_boxes(self, labels_yolo, labels_pifpaf, method):
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
            box_filtered = self.fuse_by_confidence(labels_yolo, labels_pifpaf)
        else:
            box_filtered.extend(labels_pifpaf)

        return box_filtered

    
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
            if int(box[3]) > int(box[1]) and int(box[2]) > int(box[0]):
                img[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = \
                    cv2.GaussianBlur(
                        img[int(box[1]):int(box[3]), int(box[0]):int(box[2])],
                        ksize,
                        sigmaX)


def show_results_xyxy(img, xyxys, mode="roi"):
    for xyxy in xyxys:
        h,w,_ = img.shape
        tl = 2 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
        x1 = int(xyxy[0])
        y1 = int(xyxy[1])
        x2 = int(xyxy[2])
        y2 = int(xyxy[3])
        cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    return img


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


def test_one():
    # Read the parameters
    cfg = get_cfg_defaults()
    args = parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # Initialize the anonymizer
    fa = Face_Anonymizer(cfg)

    # Go through all the images
    image_dir = cfg.IMAGE_DIR
    image_paths = sorted(os.listdir(image_dir))

    cv2.namedWindow("ROI", cv2.WND_PROP_FULLSCREEN)

    for image_path in image_paths:
        image = cv2.imread(os.path.join(image_dir, image_path))  # BGR
        assert image is not None, 'Image Not Found ' + image_path

        # Detect region of interest
        rois, lps = fa.detect_roi(image, BGR=True)

        fa.anonymize_rois(image, rois)
        
        # Visualize
        show_results_xyxy(image, rois)
        cv2.imshow("ROI", image)
        cv2.waitKey(0)


def test_two():
    # Read the parameters
    cfg = get_cfg_defaults()
    args = parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # Initialize the anonymizer
    fa = Face_Anonymizer(cfg)

    # Go through all the images
    # image_dir = cfg.IMAGE_DIR
    # image_paths = sorted(os.listdir(image_dir))
    video_path = '/home/henry/Documents/data/IMG_0465.MOV'
    cap = cv2.VideoCapture(video_path)

    cv2.namedWindow("ROI", cv2.WND_PROP_FULLSCREEN)

    idx = 0
    while(cap.isOpened()):
        ret, image = cap.read()
        idx = idx + 1
        if idx % 6 != 0 or image is None:
            continue
        # Detect region of interest
        rois, lps = fa.detect_roi(image, BGR=True)
        fa.anonymize_rois(image, rois)

        
        # Visualize
        show_results_xyxy(image, rois)
        cv2.imshow("ROI", image)
        cv2.waitKey(100)


def process_a_bag(fa, rosbag_dir):
    for folder_item in os.listdir(rosbag_dir):
        if not folder_item.startswith('avt') or folder_item.endswith('_anonymized'):
            continue

        image_dir = os.path.join(rosbag_dir, folder_item)
        image_names = sorted(os.listdir(image_dir))
        image_output_dir = os.path.join(rosbag_dir, folder_item + '_anonymized')
        if not os.path.exists(image_output_dir):
            os.mkdir(image_output_dir)

        # cv2.namedWindow("ROI", cv2.WND_PROP_FULLSCREEN)

        for image_name in image_names:
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)  # BGR
            if image is None:
                print('Image Not Found ' + image_path)
                continue

            # Detect region of interest
            rois, lps = fa.detect_roi(image, BGR=True)

            fa.anonymize_rois(image, rois)
            fa.anonymize_rois(image, lps)
            # Visualize
            # show_results_xyxy(image, rois)
            # cv2.imshow("ROI", image)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(image_output_dir, image_name), image)


def test_process_a_bag():
    # Read the parameters
    cfg = get_cfg_defaults()
    args = parse_args()
    cfg.merge_from_file("config/default_api.yaml")

    # Initialize the anonymizer
    fa = Face_Anonymizer(cfg)

    # Go through all the images
    rosbag_dir = "/home/henry/Documents/data/ros/05-13-2022/hopkins/2022-05-13-11-49-59_2"
    
    process_a_bag(fa, rosbag_dir)



if __name__ == '__main__':
    test_process_a_bag()

