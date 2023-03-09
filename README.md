# Robust Human Identity Anonymization using Pose Estimation

## Introduction

This is the offcial repository for our paper: __Robust Human Identity Anonymization using Pose Estimation__[[IEEE]](https://ieeexplore.ieee.org/document/9926568)[[arXiv]](http://arxiv.org/abs/2301.04243).

[![Watch the video](https://img.youtube.com/vi/XQaNiXgGr40/maxresdefault.jpg)](https://youtu.be/XQaNiXgGr40)


Authors: Hengyuan Zhang*, Jing-Yan Liao*, David Paz and Henrik I. Christensen.

Abstract: Many outdoor autonomous mobile platforms require more human identity anonymized data to power their data-driven algorithms. The human identity anonymization should be robust so that less manual intervention is needed, which remains a challenge for current face detection and anonymization systems. In this paper, we propose to use the skeleton generated from the state-of-the-art human pose estimation model to help localize human heads. We develop criteria to evaluate the performance and compare it with the face detection approach. We demonstrate that the proposed algorithm can reduce missed faces and thus better protect the identity information for the pedestrians. We also develop a confidence-based fusion method to further improve the performance.

__Note__: The source code in this repository are for reference and may not be fully packaged. We tried to make it more user friendly with the detect_*_api.py files.

## Structure
```
    # For experiments
    detect_roi_api.py  # major functions and fusion
    detect_face_api.py # face detector (YOLO5Face)
    detect_pose_api.py # pose detector (OpenPifPaf)
    draw_pifpaf.py     # infer head

    # For paper source code
    detect_face.py # given json of output of face detection and pose detection, fuse and track.    
    evaluate.py    # evaluate the detection
    sort.py        # tracker

    # Data processing pipeline
    utils/rosbag_to_files.py    # extract data from rosbag
    detect_roi_api.py           # generate the json file
    utils/rosbag_repack.py      # pack anonymized data into rosbag
    utils/rosbag_sampling.py    # sample for review
```

## Guide
- Install requirements
``pip install -r AVL.env.txt``
- Install weights
[face detection weight](https://drive.google.com/open?id=12O1RPth4CJR_Fk5-Izr4a466PpVxzV9R&authuser=j3liao%40ucsd.edu&usp=drive_fs) or [YOLO5Face repository pretrained models](https://github.com/deepcam-cn/yolov5-face#pretrained-models)

- If you want to use a different configuration other than specifed in the config/base_config_detect_face_api.py, please copy the default_api.yaml, modify it and pass it to the command line. For example, run: python detect_roi_api --cfg myconfig.yaml

## Cite
__Bibtex__
```
@inproceedings{zhang2022CASE:anonymization,
  address = {Mexico City},
  author = {H. Zhang and J.-Y. Liao and D. Paz and
 H. Christensen},
  booktitle = {18th International Conference on Automation Science
 and Engineering (CASE)},
  month = {August},
  organization = {IEEE},
  pages = { },
  title = {Robust Human Identity Anonymization using Pose
 Estimation},
  year = {2022}
 }
```
