""" Convert a rosbag to files for certain specified topics.

Author: Henry Zhang
Date:November 12, 2021
Reference: http://wiki.ros.org/rosbag/Cookbook
"""


from __future__ import print_function
import os
from os import path as osp
import numpy as np
import rosbag
import pypcd
import cv2
import json
import yaml
import argparse

from utils_ros import pointcloud2_to_xyz_intensity_array


def test_bag_to_files(bag_path, dir=None):
    """Convert a rosbag to files. Only certain topics are accepted.
    Place the large topics in topic_large, small topics in topic_dict
    """
    bag_name = bag_path.split("/")[-1]
    if dir is None:
        output_dir = bag_path[0:-4]
    else:
        output_dir = os.path.join(dir, bag_name[0:-4])
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    
    topic_dir_dict = {}

    # only take topics within this category
    topic_large = {
        '/points_raw', 
        '/livox/lidar', 
        '/avt_cameras/camera1/image_color/compressed',
        '/avt_cameras/camera6/image_color/compressed',
        '/avt_cameras/camera3/image_rect_color',
        '/avt_cameras/camera4/image_rect_color'
    }

    # These topics are usually small messages, we store them into a dict
    # and save all messages in each topic into a file.
    topic_dict = {
        '/tf':{},
        '/localizer_pose':{},
        '/current_velocity':{},
        '/current_pose':{},
        '/livox/imu':{},
        '/fix':{},
        '/vel':{},
    }

    print('These topics are going to be stored into folders if exist')
    for topic in topic_large: print(topic)
    print('')
    print('These topics will be stored into separate folders if exist')
    for key in topic_dict.keys(): print(key)

    bag = rosbag.Bag(bag_path)
    type = bag.get_type_and_topic_info()
    print('\n All topics in the bag')
    print(type.topics.keys())

    for topic, msg, t in bag.read_messages():
        if not topic in topic_large and not topic in topic_dict:
            # print("'{}'".format(topic),end=", ")
            continue # skip the topic
        
        header = msg.header if topic != '/tf' else msg.transforms[0].header
        timestamp_string = '{:.9f}'.format(header.stamp.to_time())

        if topic in topic_large:
            if topic in topic_dir_dict:
                topic_dir = topic_dir_dict[topic]
            else:
                # create dir for the topic
                topic_dir_name = '_'.join([item for item in topic.split('/') if item != ''])
                topic_dir = osp.join(output_dir, topic_dir_name)
                if not osp.exists(topic_dir):
                    os.mkdir(topic_dir)
                topic_dir_dict[topic] = topic_dir
        
        # For large topics, we write to the folder
        if msg._type == 'sensor_msgs/PointCloud2':
            pcd_array = pointcloud2_to_xyz_intensity_array(msg)
            pcd_new = pypcd.pypcd.make_xyz_label_point_cloud(pcd_array, label_type='f')
            pcd_new.fields[3] = 'intensity'
            filename = osp.join(topic_dir, '{}.pcd'.format(timestamp_string))
            pypcd.pypcd.save_point_cloud_bin(pcd_new, filename)
        elif msg._type == 'sensor_msgs/CompressedImage':
            image_format = 'jpg'
            np_arr = np.fromstring(msg.data, np.uint8)
            image_in = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            filename = osp.join(topic_dir, '{}.{}'.format(timestamp_string, image_format))
            cv2.imwrite(filename, image_in)
        elif msg._type == 'sensor_msgs/Image':
            image_format = 'png'
            np_arr = np.fromstring(msg.data, np.uint8)
            image_in = np_arr.reshape(msg.height, msg.width, -1)
            filename = osp.join(topic_dir, '{}.{}'.format(timestamp_string, image_format))
            cv2.imwrite(filename, image_in)
        
        # For smaller topcis, we store them in a dictinary first
        elif msg._type == 'sensor_msgs/Imu':
            topic_dict[topic][timestamp_string] = msg2json(msg)
        elif msg._type == 'geometry_msgs/TwistStamped':
            topic_dict[topic][timestamp_string] = msg2json(msg)
        elif msg._type == 'geometry_msgs/PoseStamped':
            topic_dict[topic][timestamp_string] = msg2json(msg)
        elif msg._type == 'tf2_msgs/TFMessage':
            tfs = msg.transforms[0]
            if 'livox' in tfs.child_frame_id or 'livox' in tfs.header.frame_id:
                print(tfs.child_frame_id, tfs.header.frame_id, tfs.transform.rotation)
            topic_dict[topic][timestamp_string] = msg2json(msg)
        elif msg._type == 'sensor_msgs/NavSatFix':
            topic_dict[topic][timestamp_string] = msg2json(msg)
        else:
            # print(msg._type)
            pass

    # write each small topic into a single file
    for topic in topic_dict:
        if len(topic_dict[topic].keys()) == 0:
            continue # skip empty topics

        topic_formatted_name = '_'.join([item for item in topic.split('/') if item != ''])
        filename = osp.join(output_dir, '{}.json'.format(topic_formatted_name))
        with open(filename, 'w') as fp:
            json.dump(topic_dict[topic], fp, indent=4)


def msg2json(msg):
   ''' Convert a ROS message to JSON format'''
   y = yaml.full_load(str(msg))
   return y # json.dumps(y,indent=4)


def cmdline_args():
        # Make parser object
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    
    p.add_argument("bagfile_path", help='Path to the rosbag ')
    p.add_argument("--dir", type=str, help='the directory data will be extracted to')
    return(p.parse_args())


def main():
    args = cmdline_args()
    print(args.bagfile_path)
    if args.dir is not None:
        print("output dir", args.dir)
    
    test_bag_to_files(args.bagfile_path, args.dir)


if __name__ == "__main__":
    main()