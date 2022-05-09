import cv2
import copy

import torch
import torch.backends.cudnn as cudnn

# Face Detector dependencies
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path


class Face_Detector():

    def __init__(self, cfg):
        self.image_size = cfg.IMAGE_SIZE
        self.model, self.device = self.init_model(cfg.WEIGHTS)
        self.conf_thres = cfg.CONF_THRES
        self.iou_thres = cfg.IOU_THRES


    def init_model(self, weights):
        if torch.cuda.is_available():
            print("=====GPU=====")
        else:
            print("=====CPU=====")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = attempt_load(weights, map_location=device)  # load FP32 model
        return model, device
    

    def detect(self, image, BGR=False):
        """ Given an image, return face detection."""
        img0 = copy.deepcopy(image)
        h0, w0 = image.shape[:2]  # orig hw
        r = self.image_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(self.image_size, s=self.model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]

        # # Convert
        if BGR:
            img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

        # Run inference
        # t0 = time.time()

        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)

        # Process detections
        dets = []
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(image.shape)[[1, 0, 1, 0]].to(self.device)  # normalization gain whwh
            gn_lks = torch.tensor(image.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(self.device)  # normalization gain landmarks
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                det[:, 5:15] = self.scale_coords_landmarks(img.shape[2:], det[:, 5:15], image.shape).round()
                for j in range(det.size()[0]):
                    xyxyconf = det[j, :5].cpu().numpy()
                    dets.append(xyxyconf)
                    xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                    class_num = det[j, 15].cpu().numpy()

                    det_dict = {
                        'xywh': xywh,
                        'conf': conf.tolist(),
                        'landmarks': landmarks,
                        'class_num': class_num.tolist(),
                        'xyxyconf': xyxyconf.tolist()
                    }

        return dets
    

    def scale_coords_landmarks(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
        coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
        coords[:, :10] /= gain
        #clip_coords(coords, img0_shape)
        coords[:, 0].clamp_(0, img0_shape[1])  # x1
        coords[:, 1].clamp_(0, img0_shape[0])  # y1
        coords[:, 2].clamp_(0, img0_shape[1])  # x2
        coords[:, 3].clamp_(0, img0_shape[0])  # y2
        coords[:, 4].clamp_(0, img0_shape[1])  # x3
        coords[:, 5].clamp_(0, img0_shape[0])  # y3
        coords[:, 6].clamp_(0, img0_shape[1])  # x4
        coords[:, 7].clamp_(0, img0_shape[0])  # y4
        coords[:, 8].clamp_(0, img0_shape[1])  # x5
        coords[:, 9].clamp_(0, img0_shape[0])  # y5
        return coords


def parse_args():
    """ Parse the command line arguments """

    import sys
    import argparse
    
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


def test_face_detector():
    import os
    from config.base_config_detect_face_api import get_cfg_defaults
    
    # Read the parameters
    cfg = get_cfg_defaults()
    args = parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # Initialize the anonymizer
    fd = Face_Detector(cfg.FACE_DETECTOR)

    # Go through all the images
    image_dir = cfg.IMAGE_DIR
    image_paths = sorted(os.listdir(image_dir))
    for image_path in image_paths:
        image = cv2.imread(os.path.join(image_dir, image_path))  # BGR
        assert image is not None, 'Image Not Found ' + image_path

        # Detect region of interest
        roi = fd.detect(image, BGR=True)
        print(roi)


if __name__ == '__main__':
    test_face_detector()


