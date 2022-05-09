import cv2
import openpifpaf
import torch

print('OpenPifPaf version', openpifpaf.__version__)
print('PyTorch version', torch.__version__)



class Pose_Detector():

    def __init__(self, cfg):
        self.model = openpifpaf.Predictor(checkpoint='shufflenetv2k16')


    def detect(self, image, BGR=False):
        predictions, gt_anns, image_meta = self.model.numpy_image(image)
        dets = self.format_predictions(predictions)
        return dets
    

    def format_predictions(self, predictions):
        """ Given a list of Openpifpaf predicted object, format it to json. """
        dets = []
        for pred in predictions:
            res_dict = {}
            res_dict["keypoints"] = pred.data.reshape(-1)
            res_dict["bbox"] = pred.bbox()
            res_dict["score"] = pred.score
            res_dict["category_id"] = pred.category_id
            dets.append(res_dict)
        return dets


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


def test_pose_detector():
    import os
    from config.base_config_detect_face_api import get_cfg_defaults
    
    # Read the parameters
    cfg = get_cfg_defaults()
    args = parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # Initialize the anonymizer
    pd = Pose_Detector(cfg.POSE_DETECTOR)

    # Go through all the images
    image_dir = cfg.IMAGE_DIR
    image_paths = sorted(os.listdir(image_dir))
    for image_path in image_paths:
        image = cv2.imread(os.path.join(image_dir, image_path))  # BGR
        assert image is not None, 'Image Not Found ' + image_path

        # Detect region of interest
        roi = pd.detect(image, BGR=True)
        print(roi)


if __name__ == '__main__':
    test_pose_detector()

