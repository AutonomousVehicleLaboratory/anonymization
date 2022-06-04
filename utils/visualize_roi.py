import os
import sys
import json
import argparse
import cv2

from draw_pifpaf import draw_skeleton

class AnomymizationViewer(object):
    def __init__(self):
        self.step = True
        self.window_name = "Viewer"
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        self.image_path_list = []
        self.image_idx = 0
        self.show_skeleton = True
        self.show_conf = True
        self.print_control()

    def open_image(self, image_path):
        image = cv2.imread(image_path)  # BGR
        if image is None:
            print('Image Not Found ' + image_path)
            return None
        if not image_path in self.image_path_list:
            self.image_path_list.append(image_path)
        return image


    def print_control(self):
        print("\nControl:")
        print("next new image: 'space' or 'Enter'")
        print("previous shown image: 'a' or 'p")
        print("next shown image: 'd' or 'n'")
        print("toggle skeleton: 'k'")
        print("toggle confidence: 'c'")
        print("quit: 'q'")
        print("\n")


    def show(self, image):
        cv2.imshow(self.window_name, image)
        if self.step:
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(0.1)

        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(0)
        
        if key == ord("p") or key == ord("a"):
            self.image_idx = self.image_idx + 1
            if not self.image_idx < len(self.image_path_list):
                self.image_idx = len(self.image_path_list)
            return self.image_path_list[-self.image_idx].split('/')[-1]
        
        if key == ord("n") or key == ord("d"):
            self.image_idx = self.image_idx - 1
            if not self.image_idx > 0:
                self.image_idx = 1
            return self.image_path_list[-self.image_idx].split('/')[-1]
        
        if key == ord("k"):
            self.show_skeleton = not self.show_skeleton
        if key == ord("c"):
            self.show_conf = not self.show_conf
        
        if key == ord("h"):
            self.print_control()

        self.image_idx = 0

    
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
            cv2.rectangle(img, (x1,y1), (x2, y2), color, thickness=tl, lineType=cv2.LINE_AA)

            if self.show_conf:
                cv2.putText(img, str(xyxy[4])[0:4], (x1,y1-10), 0, 0.9, color, 2, cv2.LINE_AA)

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

        image_idx = 0
        image_name = image_names[image_idx]
        while True:
            det_frame_dict = {}
            image_path = os.path.join(image_dir, image_name)
            image = viewer.open_image(image_path)
            if image is None:
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
            if viewer.show_skeleton:
                draw_skeleton(image, pose, viewer.show_conf)
            image_name = viewer.show(image)
            # cv2.imwrite(os.path.join(image_output_dir, image_name), image)

            if image_name == None:
                image_idx = image_idx + 1
                if image_idx < len(image_names):
                    image_name = image_names[image_idx]
                else:
                    break

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