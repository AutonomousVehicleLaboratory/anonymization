## Structure
```
    # For experiments
│   detect_roi_api.py  # major functions and fusion
    detect_face_api.py # face detector (YOLO5Face)
    detect_pose_api.py # pose detector (OpenPifPaf)
    draw_pifpaf.py     # infer head

    # For paper source code
    detect_face.py # given json of output of face detection and pose detection, fuse and track.
    evaluate.py    # evaluate the detection
│   sort.py        # tracker   
```

## Guide
- Install requirements
``pip install -r AVL.env.txt``
- Install weights
[face detection weight](https://drive.google.com/open?id=12O1RPth4CJR_Fk5-Izr4a466PpVxzV9R&authuser=j3liao%40ucsd.edu&usp=drive_fs)

- If you want to use a different configuration other than specifed in the config/base_config_detect_face_api.py, please copy the default_api.yaml, modify it and pass it to the command line. For example, run: python detect_face_api --cfg myconfig.yaml


## Tracker tunning
- You don't need to rerun the detection everytime when you are tunning the tracker. We provided detect_and_save, track_from_save to allow you run the detection on cluster and save the detection to a json file. Then you can download the json file to your local machine and play the tracker, even when you don't have a GPU on your local machine.
