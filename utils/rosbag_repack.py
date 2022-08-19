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


def test_bag_repack(bag_path, dir=None):
    print("processing rosbag:", bag_path)
    bag_name = bag_path.split("/")[-1]
    if dir is None:
        output_dir = bag_path[0:-4]
        outbag_path = bag_path[0:-4] + '_anonymized.bag'
    else:
        output_dir = os.path.join(dir, bag_name[0:-4])
        outbag_path = os.path.join(dir, bag_name[0:-4] + '_anonymized.bag')
    if not osp.exists(output_dir):
        print('no output dir for each bag, use args.dir instead')
        output_dir = dir

    # only take topics within this category
    topics_origin = {
        '/avt_cameras/camera3/image_rect_color',
        '/avt_cameras/camera4/image_rect_color'
    }

    topics_anony = {topic_name:topic_name + '_anonymized' for topic_name in topics_origin}

    print('These topics are going to be replaed')
    for topic in topics_origin: print(topic)
    # print(topics_anony)    
    
    topic_dir_name_dict = {}
    for topic in topics_origin:
        # create dir for the topic
        topic_dir_name = '_'.join([item for item in topic.split('/') if item != ''])
        topic_dir_name_dict[topic] = topic_dir_name
            
    # print(topic_dir_name_dict)

    det_json_dict = {}

    bridge = CvBridge()

    with rosbag.Bag(outbag_path, 'w') as outbag:
        bag = rosbag.Bag(bag_path)
        type = bag.get_type_and_topic_info()
        print('\n All topics in the bag')
        print(type.topics.keys())
        
        for topic, msg, t in bag.read_messages():
            if topic in topics_origin:
                # replace
                topic_dir_name = topic_dir_name_dict[topic]
                
                if topic not in det_json_dict:
                    det_path = os.path.join(output_dir, bag_name[0:-4] + '_' + topic_dir_name + '_det.json')
                    with open(det_path, 'r') as det_json_fp:
                        det_json = json.load(det_json_fp)
                        det_json_dict[topic] = det_json
                else:
                    det_json = det_json_dict[topic]

                timestamp_string = '{:.9f}'.format(msg.header.stamp.to_time())
                image_format = 'png'
                image_name = '{}.{}'.format(timestamp_string, image_format)
                det_frame_dict = det_json[image_name]
                roi = det_frame_dict['roi']
                lp = det_frame_dict['lp']

                np_arr = np.fromstring(msg.data, np.uint8)
                image_in = np_arr.reshape(msg.height, msg.width, -1)
                # image_in = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                anonymize_rois(image_in, roi)
                anonymize_rois(image_in, lp)
                # cv2.imshow('Image', image_in)
                # cv2.waitKey(0)
                msg_out = bridge.cv2_to_imgmsg(image_in, encoding="passthrough")
                outbag.write(topics_anony[topic], msg_out, t)

            else:
                outbag.write(topic, msg, t) # msg.header.stamp)


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
            test_bag_repack(bag_path, args.dir)
            if args.remove==True:
                print("Removing bag file:", bag_path)
                os.remove(bag_path)

if __name__ == "__main__":
    main()
