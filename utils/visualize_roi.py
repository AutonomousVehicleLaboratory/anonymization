import os
import sys
import json
import argparse
import cv2

from draw_pifpaf import draw_skeleton


class AnomymizationViewer(object):

    def __init__(self, save=False, shrink_head=False):
        self.step = True
        self.window_name = "Viewer"
        if not save:
            cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        self.image_path_list = []
        self.image_idx = 0
        self.show_skeleton = not save
        self.show_conf = not save
        self.show_roi = True
        self.apply_blur = True
        self.show_yolo = False
        self.show_openpifpaf_head = False
        self.show_lp = True
        self.shrink_head = shrink_head
        self.shrink_head_ratio = 0.8
        self.print_control()


    def open_image(self, image_path):
        print('open image:', image_path)
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
        print("jump to next folder: 'j'")
        print("toggle skeleton(k), confidence(c), roi(r), yolo face(y), openpifpaf head(o)")
        print("toggle lp(l), shrink head(s), apply blur(b)")
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
        if key == ord("r"):
            self.show_roi = not self.show_roi
        if key == ord('b'):
            self.apply_blur = not self.apply_blur
        if key == ord('y'):
            self.show_yolo = not self.show_yolo
        if key == ord('o'):
            self.show_openpifpaf_head = not self.show_openpifpaf_head
        if key == ord('l'):
            self.show_lp = not self.show_lp
        if key == ord('s'):
            self.shrink_head = not self.shrink_head
        if key == ord('j'):
            return 'jump'
        
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


    def anonymize_rois(self, image, rois):
        limit = image.shape 
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
                image[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = \
                    cv2.GaussianBlur(
                        image[int(box[1]):int(box[3]), int(box[0]):int(box[2])],
                        ksize,
                        sigmaX)


    def shrink_head_boxes(self, head_boxes):
        new_head_boxes = []
        ratio = self.shrink_head_ratio / 2 + 0.5
        for head in head_boxes:
            new_head = [
                ratio * head[0] + (1-ratio) * head[2],
                ratio * head[1] + (1-ratio) * head[3],
                (1-ratio) * head[0] + ratio * head[2],
                (1-ratio) * head[1] + ratio * head[3],
                head[4]
            ]
            new_head_boxes.append(new_head)
        
        return new_head_boxes


def process_a_dir(viewer, data_dir, anony_source=True, save=False):
    print("processing: ", data_dir)
    for folder_item in os.listdir(data_dir):
        if not folder_item.startswith('avt'):
            continue
        
        if anony_source:
            if not folder_item.endswith('anonymized'):
                continue
            det_path = os.path.join(data_dir, folder_item[0:-11] + '_det_new.json')
        else:
            if not folder_item.endswith('color'):
                continue
            det_path = os.path.join(data_dir, folder_item + '_det_new.json')

        if save:
            save_dir = os.path.join(data_dir, folder_item + '_save_box')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

        image_dir = os.path.join(data_dir, folder_item)
        image_names = sorted(os.listdir(image_dir))
        
        with open(det_path, 'r') as fp:
            det_dict = json.load(fp)
            print('detection json file loaded from ', det_path)

        image_idx = 0
        image_name = image_names[image_idx]
        while True:
            det_frame_dict = {}
            image_path = os.path.join(image_dir, image_name)
            image = viewer.open_image(image_path)
            if image is not None and image_name in det_dict:

                # print(det_frame_dict['lp'])
                det_frame_dict = det_dict[image_name]
                
                roi = det_frame_dict['roi']
                face = det_frame_dict['face']
                head = det_frame_dict['head']
                pose = det_frame_dict['pose']
                lp = det_frame_dict['lp']

                # Visualize
                if viewer.shrink_head:
                    head = viewer.shrink_head_boxes(head)
                if viewer.apply_blur:
                    viewer.anonymize_rois(image, roi)
                    viewer.anonymize_rois(image, lp)
                if viewer.show_roi:
                    viewer.show_results_xyxy(image, roi, color=(0,255,0))
                if viewer.show_yolo:
                    viewer.show_results_xyxy(image, face, color=(0,0,255))
                if viewer.show_openpifpaf_head:
                    viewer.show_results_xyxy(image, head, color=(0,255,255))
                if viewer.show_lp:
                    viewer.show_results_xyxy(image, lp, color=(255,0,0))
                if viewer.show_skeleton:
                    for p in pose:
                        draw_skeleton(image, p, viewer.show_conf)
                
                if save:
                    cv2.imwrite(os.path.join(save_dir, image_name), image)
                    image_name = None
                else:
                    viewer.set_image_title(image_path)
                    image_name = viewer.show(image)
            elif image_name not in det_dict:
                print('image not in dict:', image_name)
                image_name = None

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
                for p in pose:
                    draw_skeleton(image, p, viewer.show_conf)
            image_name = viewer.show(image)
            # cv2.imwrite(os.path.join(image_output_dir, image_name), image)

            if image_name is None or image is None:
                image_idx = image_idx + 1
                if image_idx < len(image_names):
                    image_name = image_names[image_idx]
                else:
                    break
            elif image_name == 'jump':
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
        help="process multiple extracted rosbag dir in this folder"
    )
    parser.add_argument(
        '--anony_source',
        action='store_false',
        help="use anonymized source"
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='save the visualized content'
    )
    parser.add_argument(
        '--shrink',
        action='store_true',
        help='shrink the head bounding box'
    )

    args = parser.parse_args(sys.argv[1:])
    return args


def test_process_multiple_bags():
    args = parse_args()

    if args.save==True and args.anony_source==True:
        print('Batch save only for processing original images')
        exit(0)
    
    viewer = AnomymizationViewer(save=args.save, shrink_head=args.shrink)

    if args.multiple==True:
        dir_list = sorted([item for item in os.listdir(args.dir) if not item.endswith(".bag")])
        print(dir_list)
        for dir_name in dir_list:
            data_dir = os.path.join(args.dir, dir_name)
            process_a_dir(viewer, data_dir, args.anony_source, args.save)
    else:
        data_dir = args.dir
        process_a_dir(viewer, data_dir, args.anony_source, args.save)


if __name__ == '__main__':
    test_process_multiple_bags()
