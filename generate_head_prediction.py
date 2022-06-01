# -*- coding: UTF-8 -*-
import argparse
import os
import sys
import numpy as np
import json

from config.base_config_detect_face import get_cfg_defaults
from utils.draw_pifpaf import generate_head_bbox


def predict_and_save(save_dir):
    pifpaf_path = os.path.join(save_dir, 'detection_pifpaf.json')
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
    output_path = os.path.join(save_dir, "pifpaf_pred_head_angle.json")
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