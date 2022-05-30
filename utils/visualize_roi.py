import os
import sys
import json
import argparse
import cv2


class AnomymizationViewer(object):
    def __init__(self):
        self.step = True
        self.window_name = "Viewer"
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)


    def show(self, image):
        cv2.imshow(self.window_name, image)
        if self.step:
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(0.1)

        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(0)

    
    def set_image_title(self, title):
        cv2.setWindowTitle(self.window_name, title)


    def show_results_xyxy(self, img, xyxys, color=(0,255,0)):
        for xyxy in xyxys:
            h,w,_ = img.shape
            tl = 2 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            x2 = int(xyxy[2])
            y2 = int(xyxy[3])
            cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

        return img

def process_a_dir(viewer, data_dir):
    print("processing: ", data_dir)
    for folder_item in os.listdir(data_dir):
        if not folder_item.startswith('avt') or not folder_item.endswith('anonymized'):
            continue

        image_dir = os.path.join(data_dir, folder_item)
        image_names = sorted(os.listdir(image_dir))
        det_path = os.path.join(data_dir, folder_item[0:-11] + '_det.json')
        with open(det_path, 'r') as fp:
            det_dict = json.load(fp)
            print('detection json file loaded from ', det_path)

        for image_name in image_names:
            det_frame_dict = {}
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)  # BGR
            if image is None:
                print('Image Not Found ' + image_path)
                continue

            # print(det_frame_dict['lp'])
            det_frame_dict = det_dict[image_name]
            
            roi = det_frame_dict['roi']
            face = det_frame_dict['face']
            head = det_frame_dict['head']
            pose = det_frame_dict['pose']
            lp = det_frame_dict['lp']

            # Visualize
            viewer.show_results_xyxy(image, roi, color=(0,255,0))
            viewer.show_results_xyxy(image, lp, color=(255,0,0))
            viewer.set_image_title(image_path)
            viewer.show(image)
            # cv2.imwrite(os.path.join(image_output_dir, image_name), image)


def parse_args():
    """ Parse the command line arguments """
    parser = argparse.ArgumentParser(description='visualize detection results')
    parser.add_argument(
        "--dir", 
        type=str, 
        help='the directory of extracted rosbag data')
    parser.add_argument(
        '--multiple', 
        action='store_true', 
        help="process multiple extracted rosbag dir in this folder")

    args = parser.parse_args(sys.argv[1:])
    return args


def test_process_multiple_bags():
    args = parse_args()
    viewer = AnomymizationViewer()

    if args.multiple==True:
        dir_list = sorted([item for item in os.listdir(args.dir) if not item.endswith(".bag")])
        print(dir_list)
        for dir_name in dir_list:
            data_dir = os.path.join(args.dir, dir_name)
            process_a_dir(viewer, data_dir)
    else:
        data_dir = args.dir
        process_a_dir(viewer, data_dir)


if __name__ == '__main__':
    test_process_multiple_bags()