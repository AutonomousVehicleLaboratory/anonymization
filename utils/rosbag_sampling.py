""" Create a new rosbag based on an old one.

Author: Henry Zhang
Date: August 12, 2022
"""


from __future__ import print_function
import os
from os import path as osp
from time import sleep
import numpy as np
import rosbag
import cv2
from cv_bridge import CvBridge
import json
import yaml
import argparse


def anonymize_rois(img, rois):
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


def cmdline_args():
        # Make parser object
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    
    p.add_argument("bagfile_path", help='Path to the rosbag ')
    p.add_argument("--dir", type=str, help='the directory data was extracted to')
    p.add_argument('--overwrite', action='store_true', help='skip bag if folder exist')
    p.add_argument('--remove', action='store_true', help='remove bag after extraction')
    return(p.parse_args())


def test_bag_sampling(bag_path, dir=None):
    print("Sampling rosbag:", bag_path)
    bag_name = bag_path.split("/")[-1]
    if dir is None:
        output_dir = bag_path[0:-4]
        outbag_path = bag_path[0:-4] + '_sample.bag'
    else:
        output_dir = os.path.join(dir, bag_name[0:-4])
        outbag_path = os.path.join(dir, bag_name[0:-4] + '_sample.bag')
    if not osp.exists(output_dir):
        print('no output dir for each bag, use args.dir instead')
        output_dir = dir

    with rosbag.Bag(outbag_path, 'w') as outbag:
        bag = rosbag.Bag(bag_path)
        type = bag.get_type_and_topic_info()
        print('\n All topics in the bag')
        print(type.topics.keys())

        sample_gap = 50
        topic_dict = {}
        for topic, msg, t in bag.read_messages():
            if topic not in topic_dict:
                topic_dict[topic] = 0
            
            if topic_dict[topic] % sample_gap == 0:
                outbag.write(topic, msg, t) # msg.header.stamp)
            topic_dict[topic] += 1


def main():
    args = cmdline_args()
    print(args.bagfile_path)
    if args.dir is not None:
        print("output dir", args.dir)

    if args.bagfile_path.endswith('bag'):
        # process a single bag
        test_bag_repack(args.bagfile_path, args.dir)
    else:
        # process a folder of bags
        bag_list = sorted([item for item in os.listdir(args.bagfile_path) if item.endswith('.bag')])
        print(bag_list)
        for bag_name in bag_list:
            bag_path = os.path.join(args.bagfile_path, bag_name)
            test_bag_sampling(bag_path, args.dir)
            if args.remove==True:
                print("Removing bag file:", bag_path)
                os.remove(bag_path)

if __name__ == "__main__":
    main()
