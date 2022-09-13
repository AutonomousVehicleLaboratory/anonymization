import os
import sys
import json
import glob
import argparse
from matplotlib import pyplot as plt


def parse_args():
    """ Parse the command line arguments """
    parser = argparse.ArgumentParser(description='visualize detection results')
    parser.add_argument(
        "--dir", 
        type=str, 
        help='the directory of extracted rosbag data')

    args = parser.parse_args(sys.argv[1:])
    return args


def merge_det(det_dict_all, det_dict):
    for key in det_dict:
        det_dict_all[key] = det_dict[key]


def get_plot_dict(det_dict_all, time_interval = 60):
    keys_sorted = sorted(list(det_dict_all.keys()))
    plot_dict = {}

    if len(keys_sorted) <= 0:
        exit(0)
    
    det_keys = list(det_dict_all[keys_sorted[0]].keys())

    timestamp_begin = float(keys_sorted[0][0:-4])
    plot_dict['timestamp'] = [timestamp_begin]
    
    for det_key in det_keys:
        plot_dict[det_key] = [0]

    for key in keys_sorted:
        det_frame = det_dict_all[key]
        timestamp = float(key[0:-4])
        
        while timestamp > timestamp_begin + time_interval:
            timestamp_begin = timestamp_begin + time_interval
            plot_dict['timestamp'].append(timestamp_begin)
            for det_key in det_keys:
                plot_dict[det_key].append(0)
        
        for det_key in det_frame:
            plot_dict[det_key][-1] += len(det_frame[det_key])

    # print(plot_dict)
    return plot_dict


def plot_from_dict(plot_dict):
    plt.figure()
    for key in plot_dict:
        if key != "timestamp":
            plt.plot(plot_dict['timestamp'], plot_dict[key], label=key)
    plt.legend()
    plt.title("Detection per minute for 2 cameras (10 fps)")
    plt.show()


def plot_det(det_dict_all):
    time_interval = 60 # seconds
    plot_dict = get_plot_dict(det_dict_all, time_interval=time_interval)
    plot_from_dict(plot_dict)


def test_plot_json():
    args = parse_args()

    print("dir: ", args.dir)
    pattern = os.path.join(args.dir, '*_det.json') 
    print("pattern: ", pattern)

    det_dict_all = {}

    for file_path in glob.glob(pattern):
        # print(file_path)
        bag_name = file_path.split('/')[-2]
        file_new_path = os.path.join('/'.join(file_path.split('/')[0:-1]), bag_name + '_' + file_path.split('/')[-1])
        # print(file_new_path)
        # continue
        with open(file_path, 'r') as fp:
            det_dict = json.load(fp)
            print('detection json file loaded from ', file_path)
            merge_det(det_dict_all, det_dict)
        
    plot_det(det_dict_all)


if __name__ == '__main__':
    test_plot_json()

