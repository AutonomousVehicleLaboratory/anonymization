import sys
import os
import json
import argparse
import numpy as np

from config.base_config_detect_face import get_cfg_defaults
from fusion.association import compute_distance_matrix


def format_detections(detections):
    det_xyxyconf = np.array([det['xyxyconf'] for det in detections])
    det_xyxy = det_xyxyconf[:,0:4]
    det_scores = det_xyxyconf[:,-1::]
    return det_xyxy, det_scores


def format_labels(labels):
    label_xyxy = []
    for gt in labels:
        box2d = gt['box2d']
        label_xyxy.append([box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']])
    
    return np.array(label_xyxy)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def associate_detections_to_labels(
        labels,
        detections,
        distance_threshold = 0.3,
        distance_method = '2diouxyxy'):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_labels
    """
    if(len(labels) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # distance_matrix = compute_cost_matrix_center_size(detections, labels, size_dist_ratio)
    distance_matrix = compute_distance_matrix(
        labels, 
        detections,
        method=distance_method)

    if min(distance_matrix.shape) > 0:
        a = (distance_matrix > 1 - distance_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(distance_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:, 1]):
            unmatched_detections.append(d)
    unmatched_labels = []
    for t, trk in enumerate(labels):
        if(t not in matched_indices[:, 0]):
            unmatched_labels.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(distance_matrix[m[0], m[1]] > 1 - distance_threshold):
            unmatched_detections.append(m[1])
            unmatched_labels.append(m[0])
        else:
            matches.append(m.reshape(1, 2))
    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_labels)


def evaluate_detection(
    detections, 
    labels, 
    verbose=False,
    threshold=0.3):
    tp_count = 0
    fp_count = 0
    fn_count = 0
    for name in labels:
        det_xyxy, det_scores = format_detections(detections[name])
        label_xyxy = format_labels( labels[name])
    
        matches, unmatched_dets, unmatched_labels = associate_detections_to_labels(
            label_xyxy, 
            det_xyxy,
            distance_threshold=threshold,
            distance_method="2diouxyxy")
        
        tp_count = tp_count + len(matches)
        fp_count = fp_count + len(unmatched_dets)
        fn_count = fn_count + len(unmatched_labels)

        if verbose:
            print(name)
            print('matched:')
            for match in matches:
                print('    ', label_xyxy[match[0]].astype(int), det_xyxy[match[1]])
            print('unmatched_det:')
            for det_id in unmatched_dets:
                print('    ', det_xyxy[det_id].astype(int))
            print('unmatched_labels:')
            for label_id in unmatched_labels:
                print('    ', label_xyxy[label_id].astype(int))
            print('tp:', tp_count, 'fp:', fp_count, 'fn:', fn_count)
            print(' ')
    
    print('threshold:', threshold, 'tp:', tp_count, 'fp:', fp_count, 'fn:', fn_count)


def parse_args():
    """ Parse the command line arguments """
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


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    args = parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    save_dir = cfg.SAVE_DIR

    detection_path = os.path.join(save_dir, 'detection.json')
    pifpaf_head_path = os.path.join(save_dir, 'pifpaf_pred_head.json')
    label_path = os.path.join(save_dir, 'labels.json')

    with open(detection_path) as fp:
        detections = json.load(fp)
    with open(pifpaf_head_path) as fp:
        det_pifpaf_head = json.load(fp)
    with open(label_path) as fp:
        labels = json.load(fp)
    label_array = labels['frames']
    label_dict = {label['name']: label['labels'] for label in label_array}
    
    evaluate_detection(
        detections, 
        label_dict, 
        verbose=False,
        threshold=0.3)
    evaluate_detection(
        det_pifpaf_head, 
        label_dict, 
        verbose=False,
        threshold=0.3)
    evaluate_detection(
        detections, 
        label_dict, 
        verbose=False,
        threshold=0.5)
    evaluate_detection(
        det_pifpaf_head, 
        label_dict, 
        verbose=False,
        threshold=0.5)
