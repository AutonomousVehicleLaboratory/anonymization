import os
import sys
import cv2
import argparse


def write_images_into_video(image_dir, video_path, frame_rate=10):
    output = None
    image_names = sorted(os.listdir(image_dir))
    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)  # BGR
        if image is None:
            print('Image Not Found ' + image_path)
            continue
        else:
            if output is None:
                frame_size = (image.shape[1], image.shape[0])
                print(frame_size)
                output = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),frame_rate, frame_size)
            else:
                output.write(image)
    
    output.release()


def test_write_images_into_video():
    args = parse_args()
    image_path = args.dir
    video_path = os.path.join(image_path, "video.mp4")
    write_images_into_video(image_path, video_path)


def parse_args():
    """ Parse the command line arguments """
    parser = argparse.ArgumentParser(description='cam_lidar_calibration')
    parser.add_argument(
        "--dir", 
        type=str, 
        help='the directory of extracted rosbag data')
    args = parser.parse_args(sys.argv[1:])
    return args


if __name__ == '__main__':
    test_write_images_into_video()
