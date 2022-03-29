import sys
import os
import json
import argparse
from tabnanny import verbose
import numpy as np
from collections import defaultdict
import cv2

from config.base_config_detect_face import get_cfg_defaults
from fusion.association import compute_distance_matrix


def format_detections(detections):
    det_xyxyconf = np.array([det['xyxyconf'] for det in detections])
    if len(det_xyxyconf) == 0:
        return np.zeros((4,0)), np.zeros((1,0))
        
    det_xyxy = det_xyxyconf[:,0:4]
    det_scores = det_xyxyconf[:,-1::]
    return det_xyxy, det_scores


def format_labels(labels):
    label_xyxy = []
    for gt in labels:
        box2d = gt['box2d']
        label_xyxy.append([box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']])
    
    return np.array(label_xyxy)


def format_two_labels(labels):
    label_face_xyxy = []
    label_head_xyxy = []
    for gt in labels:
        box2d = gt['box2d']
        if gt['category'] == 'face':
            label_face_xyxy.append([box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']])
        if gt['category'] == 'head':
            label_head_xyxy.append([box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']])
    
    return np.array(label_face_xyxy), np.array(label_head_xyxy)


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

    if distance_matrix is not None and distance_matrix.size > 0:
        a = (distance_matrix < 1 - distance_threshold).astype(np.int32)
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


def draw_labels(img, labels, thickness=1):
    for label in labels:
        color = (0,255,0) if label['category'] == "face" else (0,0,255)
        box = label  ["box2d"]
        cv2.rectangle(
            img, 
            (int(box['x1']), int(box['y1'])),
            (int(box['x2']), int(box['y2'])),
            color,
            thickness=thickness)

def draw_detections(img, detections, thickness=2):
    for det in detections:
        color = (0,255,0)
        box = det['xyxyconf']
        cv2.rectangle(
            img, 
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color,
            thickness=thickness)


def anonymize_detections(img, detections):
    ksize = (5,5)
    sigmaX = cv2.BORDER_DEFAULT                   
    for det in detections:
        color = (0,255,0)
        box = det['xyxyconf']
        img[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = \
            cv2.GaussianBlur(
                img[int(box[1]):int(box[3]), int(box[0]):int(box[2])],
                ksize,
                sigmaX)


def filter_small_objects(
    label_dict, 
    detections,
    det_pifpaf_head,
    tracking_pifpaf_head,
    image_dir, 
    threshold_max_dim, 
    viz=False):
    """ Filter small detection as well as groundtruth labels.
    The filtering is based on the maximum dimension. """
    if viz:
        cv2.namedWindow("Label", cv2.WND_PROP_FULLSCREEN)

    for name in sorted(label_dict.keys()):
        # filter labels
        labels = label_dict[name]
        new_labels = []
        for label in labels:
            if label['category'] == 'face':
                att = label['attributes']
                if "Hidden" in att and att["Hidden"] == True:
                    pass
                else:
                    new_labels.append(label) # not filter face labels
            elif label['category'] == 'head':
                box = label['box2d']
                max_dim = max((box['x2']-box['x1']), 
                              (box['y2']-box['y1']))
                if max_dim > threshold_max_dim:
                    new_labels.append(label)
        label_dict[name] = new_labels

        # Filter YOLOv5 Detection
        # dets = detections[name]
        # new_dets = []
        # for det in dets:
        #     box = det['xyxyconf']
        #     max_dim = max((box[2] - box[0]), (box[3] - box[1]))
        #     if max_dim > threshold_max_dim:
        #         new_dets.append(det)
        # detections[name] = new_dets

        # Filter OpenPifPaf detection
        # dets = det_pifpaf_head[name]
        # new_dets = []
        # for det in dets:
        #     box = det['xyxyconf']
        #     max_dim = max((box[2] - box[0]), (box[3] - box[1]))
        #     if max_dim > threshold_max_dim-10:
        #         new_dets.append(det)
        # det_pifpaf_head[name] = new_dets
    
        # Filter OpenPifPaf Head Tracking by size
        trks = tracking_pifpaf_head[name]
        new_trks = []
        for trk in trks:
            box = trk['xyxyconf']
            max_dim = max((box[2] - box[0]), (box[3] - box[1]))
            if max_dim > threshold_max_dim:
                new_trks.append(trk)
        tracking_pifpaf_head[name] = new_trks        

        if viz:
            image_path = os.path.join(image_dir, name)
            img = cv2.imread(image_path)
            draw_labels(img, new_labels)
            draw_detections(dets)
            cv2.imshow("Label", img)
            cv2.waitKey(0)
    
    if viz:
        cv2.destroyAllWindows()


def get_ratio(xyxy1, xyxy2):
    """ Get the ratio of (A and B) / B for bounding boxes A and B
    """
    xyxy = np.array([xyxy1, xyxy2])
    xyxymin = np.min(xyxy, axis=0)
    xyxymax = np.max(xyxy, axis=0)
    intersection = (xyxymin[2] - xyxymax[0]) * \
                   (xyxymin[3] - xyxymax[1])
    S2 = (xyxy2[2] - xyxy2[0]) * (xyxy2[3] - xyxy2[1])
    ratio = intersection / S2
    if ratio > 1:
        print("Error: ratio > 1 in get ratio")
    return ratio


def evaluate_two_labels(
    detections,
    labels,
    threshold_face=0.5,
    threshold_head=0.7,
    image_dir=None,
    viz=False,
    verbose=False
):
    if viz:
        cv2.namedWindow("Label", cv2.WND_PROP_FULLSCREEN)

    tp_count, fp_count, fn_count = 0, 0, 0
    afh_mfh_count = 0
    afh_mf_count = 0
    afh_mh_count = 0
    afh_mn_count = 0 # detection, associated with head and face, but not matched
    ah_mh_count = 0
    ah_mn_count = 0
    an_count = 0
    nfh_count = 0 # no detection, but head has face
    nh_count = 0 # no detection, and head has no face

    for name in labels:
        dets_xyxy, det_scores = format_detections(detections[name])
        label_face_xyxy, label_head_xyxy = format_two_labels( labels[name])

        # match detection with head labels
        matches_det_head, unmatched_dets, unmatched_heads_det = associate_detections_to_labels(
            label_head_xyxy, 
            dets_xyxy,
            distance_threshold=0.1,
            distance_method="2diouxyxy")

        # match face label with head labels
        matches_face_head, unmatched_faces, unmatched_heads_face = associate_detections_to_labels(
            label_head_xyxy, 
            label_face_xyxy,
            distance_threshold=0.1,
            distance_method="2diouxyxy")
        
        head_to_face_dict = {}
        for (head_idx, face_idx) in matches_face_head:
            head_to_face_dict[head_idx] = label_face_xyxy[face_idx]

        for (head_idx, det_idx) in matches_det_head:
            head_xyxy = label_head_xyxy[head_idx]
            det_xyxy = dets_xyxy[det_idx]
            head_ratio = get_ratio(head_xyxy, det_xyxy)
            if head_idx in head_to_face_dict:
                face_xyxy = head_to_face_dict[head_idx]
                face_ratio = get_ratio(det_xyxy, face_xyxy)
                if face_ratio > threshold_face and head_ratio > threshold_head:
                    tp_count = tp_count + 1
                    afh_mfh_count = afh_mfh_count + 1
                elif face_ratio > threshold_face:
                    afh_mf_count = afh_mf_count + 1
                elif head_ratio > threshold_head:
                    afh_mh_count = afh_mh_count + 1
                else:
                    afh_mn_count = afh_mn_count + 1
                    fp_count = fp_count + 1
                    fn_count = fn_count + 1
            else:
                if head_ratio > threshold_head:
                    ah_mh_count = ah_mh_count + 1
                else:
                    ah_mn_count = ah_mn_count + 1
                # When face doesn't exist in the label for a corresponding head,
                # We can ignore the head detection
        
        # If a head is not associated with detection,
        # It is considerered wrong only when it has a face
        for head_idx in unmatched_heads_det:
            if head_idx in head_to_face_dict:
                fn_count = fn_count + 1
                nfh_count = nfh_count + 1
            else:
                nh_count = nh_count + 1

        fp_count = fp_count + len(unmatched_dets)
        an_count = an_count + len(unmatched_dets)

        if verbose:
            print(
                # 'tp:', tp_count, 
                # 'fp:', fp_count, 
                # 'fn:', fn_count,
                'afh_mfh', afh_mfh_count,
                'afh_mf', afh_mf_count,
                'afh_mh', afh_mh_count,
                'afh_mn', afh_mn_count,
                'ah_mh', ah_mh_count,
                'ah_mn', ah_mn_count,
                'an', an_count,
                'nfh', nfh_count,
                'nh', nh_count)
        # if len(unmatched_faces) > 0:
            # print("Error: A face label doesn't come with head label")
        if viz:# and name=='1635293319.560385704.jpg':
            image_path = os.path.join(image_dir, name)
            img = cv2.imread(image_path)
            # draw_labels(img, labels[name])
            # draw_detections(img, detections[name])
            anonymize_detections(img, detections[name])
            cv2.imshow("Label", img) #[625:840, 475:740])
            cv2.waitKey(0)
            print(" ")
    
    if viz:
        cv2.destroyAllWindows()
    
    # print('thres_face:', threshold_face,
    #       'thres_head:', threshold_head,
    #     #   'tp:', tp_count, 
    #     #   'fp:', fp_count, 
    #     #   'fn:', fn_count,
    #       'afh_mfh', afh_mfh_count,
    #       'afh_mf', afh_mf_count,
    #       'afh_mh', afh_mh_count,
    #       'afh_mn', afh_mn_count,
    #       'ah_mh', ah_mh_count,
    #       'ah_mn', ah_mn_count,
    #       'an', an_count,
    #       'nfh', nfh_count,
    #       'nh', nh_count,
    #       'mn', nfh_count + afh_mn_count + ah_mn_count)
    show_missing_only = False
    if show_missing_only:
        missing_count = afh_mn_count+nfh_count+afh_mh_count
        total_count = missing_count + afh_mfh_count + afh_mf_count
        missing_rate = missing_count / total_count
        print(round(missing_rate*100, 1), end=' ')
    else:
        # pass
        # print('&', afh_mfh_count,
        #     '&', afh_mf_count,
        #     '&', afh_mh_count,
        #     '&', afh_mn_count + nfh_count,
        #     '&', ah_mh_count,
        #     '&', ah_mn_count + nh_count,
        #     '&', an_count + afh_mn_count + ah_mn_count,
        #     '\\\\')
        print(afh_mfh_count, 
              afh_mf_count,
              afh_mh_count,
              afh_mn_count + nfh_count, end=' ')



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


def evaluate_two_labels_double(
    detections_yolo,
    detections_pifpaf,
    labels,
    threshold_face=0.5,
    threshold_head=0.7,
    image_dir=None,
    viz=False,
    verbose=False
):
    if viz:
        cv2.namedWindow("Label", cv2.WND_PROP_FULLSCREEN)

    tp_count, fp_count, fn_count = 0, 0, 0
    afh_mfh_count = 0
    afh_mf_count = 0
    afh_mh_count = 0
    afh_mn_count = 0 # detection, associated with head and face, but not matched
    ah_mh_count = 0
    ah_mn_count = 0
    an_count = 0
    nfh_count = 0 # no detection, but head has face
    nh_count = 0 # no detection, and head has no face

    for name in labels:
        dets_yolo_xyxy, det_yolo_scores = format_detections(detections_yolo[name])
        dets_pifpaf_xyxy, det_pifpaf_scores = format_detections(detections_pifpaf[name])
        label_face_xyxy, label_head_xyxy = format_two_labels( labels[name])

        # match detection with head labels
        matches_det_yolo_head, unmatched_yolo_dets, unmatched_heads_det_yolo = associate_detections_to_labels(
            label_head_xyxy, 
            dets_yolo_xyxy,
            distance_threshold=0.1,
            distance_method="2diouxyxy")
        
        matches_det_pifpaf_head, unmatched_pifpaf_dets, unmatched_heads_det_pifpaf = associate_detections_to_labels(
            label_head_xyxy, 
            dets_pifpaf_xyxy,
            distance_threshold=0.1,
            distance_method="2diouxyxy")

        # match face label with head labels
        matches_face_head, unmatched_faces, unmatched_heads_face = associate_detections_to_labels(
            label_head_xyxy, 
            label_face_xyxy,
            distance_threshold=0.1,
            distance_method="2diouxyxy")
        
        head_to_face_dict = {}
        for (head_idx, face_idx) in matches_face_head:
            head_to_face_dict[head_idx] = label_face_xyxy[face_idx]

        for (head_idx, det_idx) in matches_det_yolo_head:
            head_xyxy = label_head_xyxy[head_idx]
            det_xyxy = dets_yolo_xyxy[det_idx]
            head_ratio = get_ratio(head_xyxy, det_xyxy)
            if head_idx in head_to_face_dict:
                face_xyxy = head_to_face_dict[head_idx]
                face_ratio = get_ratio(det_xyxy, face_xyxy)
                if face_ratio > threshold_face and head_ratio > threshold_head:
                    tp_count = tp_count + 1
                    afh_mfh_count = afh_mfh_count + 1
                elif face_ratio > threshold_face:
                    afh_mf_count = afh_mf_count + 1
                elif head_ratio > threshold_head:
                    afh_mh_count = afh_mh_count + 1
                else:
                    afh_mn_count = afh_mn_count + 1
                    fp_count = fp_count + 1
                    fn_count = fn_count + 1
            else:
                if head_ratio > threshold_head:
                    ah_mh_count = ah_mh_count + 1
                else:
                    ah_mn_count = ah_mn_count + 1
                # When face doesn't exist in the label for a corresponding head,
                # We can ignore the head detection
        
        # If a head is not associated with detection,
        # It is considerered wrong only when it has a face
        for head_idx in unmatched_heads_det:
            if head_idx in head_to_face_dict:
                fn_count = fn_count + 1
                nfh_count = nfh_count + 1
            else:
                nh_count = nh_count + 1

        fp_count = fp_count + len(unmatched_dets)
        an_count = an_count + len(unmatched_dets)

        if verbose:
            print(
                # 'tp:', tp_count, 
                # 'fp:', fp_count, 
                # 'fn:', fn_count,
                'afh_mfh', afh_mfh_count,
                'afh_mf', afh_mf_count,
                'afh_mh', afh_mh_count,
                'afh_mn', afh_mn_count,
                'ah_mh', ah_mh_count,
                'ah_mn', ah_mn_count,
                'an', an_count,
                'nfh', nfh_count,
                'nh', nh_count)
        # if len(unmatched_faces) > 0:
            # print("Error: A face label doesn't come with head label")
        if viz:
            image_path = os.path.join(image_dir, name)
            img = cv2.imread(image_path)
            draw_labels(img, labels[name])
            draw_detections(img, detections[name])
            cv2.imshow("Label", img)
            cv2.waitKey(0)
    
    if viz:
        cv2.destroyAllWindows()
    
    # print('thres_face:', threshold_face,
    #       'thres_head:', threshold_head,
    #     #   'tp:', tp_count, 
    #     #   'fp:', fp_count, 
    #     #   'fn:', fn_count,
    #       'afh_mfh', afh_mfh_count,
    #       'afh_mf', afh_mf_count,
    #       'afh_mh', afh_mh_count,
    #       'afh_mn', afh_mn_count,
    #       'ah_mh', ah_mh_count,
    #       'ah_mn', ah_mn_count,
    #       'an', an_count,
    #       'nfh', nfh_count,
    #       'nh', nh_count,
    #       'mn', nfh_count + afh_mn_count + ah_mn_count)
    print('&', afh_mfh_count,
          '&', afh_mf_count,
          '&', afh_mh_count,
          '&', afh_mn_count + nfh_count,
          '&', ah_mh_count,
          '&', ah_mn_count + nh_count,
          '&', an_count + afh_mn_count + ah_mn_count,
          '\\\\')


def fuse_by_confidence(labels_yolo, labels_pifpaf):
    box_filtered = []
    for label_yolo in labels_yolo:
        box_yolo = label_yolo['xyxyconf']
        in_a_box_and_low_conf = False
        for label_pifpaf in labels_pifpaf:
            box_pifpaf = label_pifpaf['xyxyconf']
            if box_yolo[0] >= box_pifpaf[0] and \
            box_yolo[2] <= box_pifpaf[2] and \
            box_yolo[1] >= box_pifpaf[1] and \
            box_yolo[3] <= box_pifpaf[3]:
                if box_yolo[-1] < box_pifpaf[-1]:
                    in_a_box_and_low_conf = True
                    break
        if not in_a_box_and_low_conf:
            box_filtered.append(label_yolo)
    for label_pifpaf in labels_pifpaf:
        box_pifpaf = label_pifpaf['xyxyconf']
        has_a_box_and_low_conf = False
        for label_yolo in labels_yolo:
            box_yolo = label_yolo['xyxyconf']
            if box_yolo[0] >= box_pifpaf[0] and \
            box_yolo[2] <= box_pifpaf[2] and \
            box_yolo[1] >= box_pifpaf[1] and \
            box_yolo[3] <= box_pifpaf[3]:
                if box_yolo[-1] > box_pifpaf[-1]:
                    has_a_box_and_low_conf = True
                    break
        if not has_a_box_and_low_conf:
            box_filtered.append(label_pifpaf)
    return box_filtered

def filter_boxes(labels_yolo, labels_pifpaf, method):
    box_filtered = []
    if method == 'remove_face':
        for label_yolo in labels_yolo:
            box_yolo = label_yolo['xyxyconf']
            in_a_box = False
            for label_pifpaf in labels_pifpaf:
                box_pifpaf = label_pifpaf['xyxyconf']
                if box_yolo[0] >= box_pifpaf[0] and \
                box_yolo[2] <= box_pifpaf[2] and \
                box_yolo[1] >= box_pifpaf[1] and \
                box_yolo[3] <= box_pifpaf[3]:
                    in_a_box = True
                    break
            if not in_a_box:
                box_filtered.append(label_yolo)
    elif method == 'remove_head':
        for label_pifpaf in labels_pifpaf:
            box_pifpaf = label_pifpaf['xyxyconf']
            has_a_box = False
            for label_yolo in labels_yolo:
                box_yolo = label_yolo['xyxyconf']
                if box_yolo[0] >= box_pifpaf[0] and \
                box_yolo[2] <= box_pifpaf[2] and \
                box_yolo[1] >= box_pifpaf[1] and \
                box_yolo[3] <= box_pifpaf[3]:
                    has_a_box = True
                    break
            if not has_a_box:
                box_filtered.append(label_pifpaf)
    elif method == 'conf_fusion':
        box_filtered = fuse_by_confidence(labels_yolo, labels_pifpaf)
    else:
        box_filtered.extend(labels_pifpaf)

    return box_filtered



def merge_detection(det_yolo, det_pifpaf, method='remove_face'):
    detection_merged = {}
    for name in det_pifpaf:
        detection_merged[name] = []
        if method == 'remove_face':
            detection_merged[name].extend(det_pifpaf[name])
            if name in det_yolo:
                filtered_boxes = filter_boxes( det_yolo[name], det_pifpaf[name], method)
                detection_merged[name].extend(filtered_boxes)
        elif method == 'conf_fusion':
            detection_merged[name] = filter_boxes( det_yolo[name], det_pifpaf[name], method)
        else:
            detection_merged[name].extend(det_yolo[name])
            if name in det_pifpaf:
                filtered_boxes = filter_boxes( det_yolo[name], det_pifpaf[name], method)
                detection_merged[name].extend(filtered_boxes) 
    return detection_merged

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    args = parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    save_dir = cfg.SAVE_DIR
    image_dir = cfg.IMAGE
 
    detection_path = os.path.join(save_dir, 'detection.json')
    pifpaf_head_path = os.path.join(save_dir, 'pifpaf_pred_head.json')
    label_face_path = os.path.join(save_dir, 'label_face.json')
    label_head_path = os.path.join(save_dir, 'label_head.json')
    tracking_yolo_face_path = os.path.join(save_dir, 'tracking_yolo_face.json')
    tracking_pifpaf_head_path = os.path.join(save_dir, 'tracking_pifpaf_head.json')

    with open(detection_path) as fp:
        detections = json.load(fp)
    with open(pifpaf_head_path) as fp:
        det_pifpaf_head = json.load(fp)
    with open(tracking_pifpaf_head_path) as fp:
        tracking_pifpaf_head = json.load(fp)
    with open(tracking_yolo_face_path) as fp:
        tracking_yolo_face = json.load(fp)
    label_dict = defaultdict(list)
    with open(label_face_path) as fp:
        label_face = json.load(fp)
        label_face_array = label_face['frames']
        for label in label_face_array:
            label_dict[label["name"]].extend(label["labels"])
    with open(label_head_path) as fp:
        label_head = json.load(fp)
        label_head_array = label_head["frames"]
        for label in label_head_array:
            label_dict[label["name"]].extend(label["labels"])
    
    threshold_max_dim = None
    if threshold_max_dim is not None:
        filter_small_objects(
            label_dict, 
            detections,
            det_pifpaf_head,
            tracking_pifpaf_head,
            image_dir, 
            threshold_max_dim=threshold_max_dim,
            viz=False)
        print(threshold_max_dim, end=' ')

    viz = False
    verbose = False 
    for i in np.arange(0.1, 1, 0.1):
        print(i, end=' ')
        threshold_face = 0.5
        threshold_head = i

        evaluate_two_labels(
            detections,
            label_dict,
            threshold_face=threshold_face,
            threshold_head=threshold_head,
            image_dir=image_dir,
            viz=viz,
            verbose=verbose
        )
        evaluate_two_labels(
            det_pifpaf_head,
            label_dict,
            threshold_face=threshold_face,
            threshold_head=threshold_head,
            image_dir=image_dir,
            viz=viz,         
            verbose=verbose
        )

        # detection_merged = merge_detection(
        #     detections, 
        #     det_pifpaf_head,
        #     method='remove_face')
        # evaluate_two_labels(
        #     detection_merged,
        #     label_dict,
        #     threshold_face=threshold_face,
        #     threshold_head=threshold_head,
        #     image_dir=image_dir,
        #     viz=viz,
        #     verbose=verbose
        # )

        # detection_merged = merge_detection(
        #     detections, 
        #     det_pifpaf_head,
        #     method='remove_head')
        # evaluate_two_labels(
        #     detection_merged,
        #     label_dict,
        #     threshold_face=threshold_face,
        #     threshold_head=threshold_head,
        #     image_dir=image_dir,
        #     viz=viz,
        #     verbose=verbose
        # )

        # detection_merged = merge_detection(
        #     detections, 
        #     det_pifpaf_head,
        #     method='')
        # evaluate_two_labels(
        #     detection_merged,
        #     label_dict,
        #     threshold_face=threshold_face,
        #     threshold_head=threshold_head,
        #     image_dir=image_dir,
        #     viz=viz,
        #     verbose=verbose
        # )
        detection_merged = merge_detection(
            detections, 
            det_pifpaf_head,
            method='conf_fusion')
        evaluate_two_labels(
            detection_merged,
            label_dict,
            threshold_face=threshold_face,
            threshold_head=threshold_head,
            image_dir=image_dir,
            viz=True,
            verbose=verbose
        )

        # evaluate_two_labels(
        #     tracking_yolo_face,
        #     label_dict,
        #     threshold_face=threshold_face,
        #     threshold_head=threshold_head,
        #     image_dir=image_dir,
        #     viz=viz,
        #     verbose=verbose
        # )
        # evaluate_two_labels(
        #     tracking_pifpaf_head,
        #     label_dict,
        #     threshold_face=threshold_face,
        #     threshold_head=threshold_head,
        #     image_dir=image_dir,
        #     viz=viz,
        #     verbose=verbose
        # )
        print(' ')
