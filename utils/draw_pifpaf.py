import numpy as np
import cv2


general_keyareas = {}
general_keyareas["face"] = [i for i in range(5)]
general_keyareas["shoulder"] = [5,6]
general_keyareas["hip"] = [11,12]
general_keyareas["knee"] = [13,14]
general_keyareas['ankle'] = [15,16]
general_keyareas['eye'] = [2,3]
general_keyareas['torso'] = [5,6,11,12]
general_keyareas['right_body'] = [2, 4,6,8,10,12,14,16]
general_keyareas['left_body'] = [1,3,5,7,9,11,13,15]

KINEMATIC_TREE_SKELETON = [
    (1, 2), (2, 4),  # left head
    (1, 3), (3, 5),
    (1, 6),
    (6, 8), (8, 10),  # left arm
    (1, 7),
    (7, 9), (9, 11),  # right arm
    (6, 12), (12, 14), (14, 16),  # left side
    (7, 13), (13, 15), (15, 17),
]

COCO_KEYPOINTS = [
    'nose',            # 0
    'left_eye',        # 1
    'right_eye',       # 2
    'left_ear',        # 3
    'right_ear',       # 4
    'left_shoulder',   # 5
    'right_shoulder',  # 6
    'left_elbow',      # 7
    'right_elbow',     # 8
    'left_wrist',      # 9
    'right_wrist',     # 10
    'left_hip',        # 11
    'right_hip',       # 12
    'left_knee',       # 13
    'right_knee',      # 14
    'left_ankle',      # 15
    'right_ankle',     # 16
]


def draw_skeleton(img, pifpaf_keypoints, show_conf=False):
    color_left = (0, 255, 255)
    color_right = (255, 0, 255)
    # openpifpaf keypoints format: (x, y, confidence)
    pp_kps = np.array(pifpaf_keypoints).reshape(-1,3)
    if len(pp_kps) == 0:
        return img
    # draw skeleton by connecting different keypoint by coco default
    for pair in KINEMATIC_TREE_SKELETON:
        partA = pair[0] -1
        partB = pair[1] -1
        # left
        color = color_left
        # right
        if partA % 2 ==0 and partB % 2 == 0:
            color = color_right
        # if confidence is not zero, the keypoint exist, otherwise the keypoint would be at (0,0)
        if  not np.isclose(pp_kps[partA, 2],  0) and not np.isclose(pp_kps[partB, 2],  0):
            cv2.line(img, pp_kps[partA,:2].astype(int), pp_kps[partB,:2].astype(int), color, 2)
    
    if show_conf:
        for kp_idx, kp in enumerate(pp_kps):
            if kp[2] > 0:
                cv2.putText(
                    img, 
                    str(kp[2])[0:4], 
                    (int(max(0, kp[0]-120*(kp_idx % 2) + 30)),
                    int(max(kp[1]-10, 0))), 
                    0, 0.9, color, 2, cv2.LINE_AA)

    return img


def predict_and_draw_head(img, pifpaf_keypoints):
    pp_kps = pifpaf_keypoints.reshape(-1,3)
    box, box_from_face, conf = generate_head_bbox(pp_kps)
    if box is not None:
        color = (0, 0, 255) if box_from_face else (255, 255, 0)
        cv2.rectangle(img, box[0], box[1], color, 1)
    

# def draw_skeleton_and_head(img, pifpaf_keypoints, predict_head):
#     # openpifpaf keypoints format: (x, y, confidence)
#     # draw skeleton by connecting different keypoint by coco default
#     draw_skeleton(img, pifpaf_keypoints)
#     if predict_head:
#         predict_and_draw_head(img, pifpaf_keypoints)


def face_to_us(pp_kps):
    left_x = get_joint_coor("left_body", pp_kps)[0]
    right_x = get_joint_coor("right_body", pp_kps)[0]
    if left_x < right_x:
        return False
    return True


def get_body_bbox(pp_kps):
    min_x, min_y, max_x, max_y = 1000, 1000, 0, 0
    for i in pp_kps:
        if np.isclose(i[2], 0):
            pass
        else:
            if i[0] < min_x:
                min_x = i[0]
            if i [0] > max_x:
                max_x = i[0]
            if i[1] < min_y:
                min_x = i[1]
            if i [1] > max_y:
                max_x = i[1]
    return np.array([min_x, min_y, max_x, max_y])

def get_joint_coor(joint_name, pp_kps):
    res_x, res_y, conf = 0, 0, 0
    count = 0
    for i in general_keyareas[joint_name]:
        if (np.isclose(pp_kps[i,2], 0)):
            pass
        else:
            res_x += pp_kps[i,0]
            res_y += pp_kps[i,1]
            conf += pp_kps[i,2]
            count += 1
    if count == 0:
        return np.array([0,0])
    res_x = int(res_x/count)
    res_y = int(res_y/count)
    conf /= count
    return np.array([res_x, res_y, conf])

def joint_exist(joint_name, pp_kps, all_exist=False):
    # general_terms = ["face", "shoulder", "hip", "knee", "ankle", "eye"]
    if joint_name in general_keyareas.keys():
        num_limit = len(general_keyareas[joint_name])//2
        if all_exist:
            num_limit = 0
        if np.sum(np.isclose(pp_kps[general_keyareas[joint_name], 2], np.zeros(len(general_keyareas[joint_name])))) > num_limit:
            return False
    else:
        index_joint = COCO_KEYPOINTS.index(joint_name)
        if np.isclose(pp_kps[index_joint, 2], 0):
            return False
    return True
# change to voting method
def get_human_height(pp_kps):
    n_s_to_h_ratio, torso_to_h_ratio, hip_knee_to_h_ratio, knee_ankle_to_h_ratio = 0.12, 0.3,  0.25, 0.25
    predicted_height = []
    height = 0
    if joint_exist("nose", pp_kps) and joint_exist("shoulder", pp_kps):
        nose2shoulder = np.linalg.norm(get_joint_coor("shoulder", pp_kps) - pp_kps[0,:2])
        predicted_height.append(nose2shoulder/n_s_to_h_ratio)
    if joint_exist("hip", pp_kps) and joint_exist("shoulder", pp_kps):
        shoulder2hip = np.linalg.norm(get_joint_coor("hip", pp_kps) - get_joint_coor("shoulder", pp_kps))
        predicted_height.append(shoulder2hip / torso_to_h_ratio)
    if joint_exist("hip", pp_kps) and joint_exist("knee", pp_kps):
        hip2knee = np.linalg.norm(get_joint_coor("knee", pp_kps) - get_joint_coor("hip", pp_kps))
        predicted_height.append(hip2knee / hip_knee_to_h_ratio)
    if joint_exist("ankle", pp_kps) and joint_exist("knee", pp_kps):
        knee2ankle = np.linalg.norm(get_joint_coor("ankle", pp_kps) - get_joint_coor("knee", pp_kps))
        predicted_height.append(knee2ankle / knee_ankle_to_h_ratio)
    # print(predicted_height)
    if predicted_height:
        height = np.median(np.array(predicted_height))
    return height

def get_ortho(a_2d):
    res = np.array([a_2d[0], -1*a_2d[1]])
    res /= np.linalg.norm(res)
    if res[1] > 0:
        res *= -1
    return res

def y_rot_mat(ang):
    return np.array([[np.cos(ang), 0., np.sin(ang)], [0.,1.,0.], [-np.sin(ang), 0., np.cos(ang)]])

def generate_head_bbox(pp_kps, shrink_ratio = 1.0):
    torso_length_head_width_ratio = 2/5*shrink_ratio
    neck_to_head_height_ratio = 1/4
    head_aspect_ratio = 1.2
    shoulder_torso_ratio = 1
    shoulder_height_ratio = 0.23
    shoulder_head_width_ratio = 1.5
    pred_human_height_head_height_ratio = 5.5
    
    if joint_exist("face", pp_kps)  and joint_exist("shoulder", pp_kps):
        head_width = 0
        head_middle_coor = get_joint_coor('face', pp_kps)[:2]
        conf = 0
        
        if joint_exist("hip", pp_kps):
            head_width = (get_joint_coor("hip",pp_kps)[1] - get_joint_coor("shoulder",pp_kps)[1])*torso_length_head_width_ratio
            conf = (get_joint_coor("hip",pp_kps)[2] + get_joint_coor("shoulder",pp_kps)[2]) / 2
        elif joint_exist("shoulder", pp_kps, all_exist=True):
            head_width = (np.amax(pp_kps[5:7, 0]) - np.amin(pp_kps[5:7, 0])) / shoulder_head_width_ratio
            conf = np.mean(pp_kps[5:7,2])
        else:
            head_width = get_joint_coor("shoulder", pp_kps)[1] - get_joint_coor("face", pp_kps)[1]
            conf = (get_joint_coor("shoulder", pp_kps)[2] + get_joint_coor("face", pp_kps)[2])/2
        
        head_bbox_x1, head_bbox_x2 = int(head_middle_coor[0] - head_width/2), int(head_middle_coor[0] + head_width/2)
        head_bbox_y1, head_bbox_y2 = int(head_middle_coor[1] - head_width * head_aspect_ratio/2), int(head_middle_coor[1] + head_width * head_aspect_ratio/2)
        box = ((head_bbox_x1, head_bbox_y1), (head_bbox_x2, head_bbox_y2))
        box_from_face = True
    elif joint_exist("shoulder", pp_kps, all_exist=True):
        shoulder_center = get_joint_coor("shoulder", pp_kps)
        head_width  = (np.amax(pp_kps[5:7, 0]) - np.amin(pp_kps[5:7, 0])) / shoulder_head_width_ratio
        head_height = head_width * head_aspect_ratio
        conf = shoulder_center[2]
        
        #find vector orthogonal to shoulder vector
        shoulder_vec = pp_kps[5,:2] - pp_kps[6,:2]
        ortho_vec = get_ortho(shoulder_vec)
        # calculate shoulder length ratio
        normal_shoulder_length = 1
        shown_shoulder_length = np.linalg.norm(shoulder_vec)
        
        if joint_exist("hip", pp_kps):
            ortho_vec = (shoulder_center - get_joint_coor("hip",pp_kps))[:2]
            torso_length = np.linalg.norm(ortho_vec)
            normal_shoulder_length = shoulder_torso_ratio * torso_length
            ortho_vec /= torso_length
            head_width = torso_length * torso_length_head_width_ratio
            conf = (get_joint_coor("hip",pp_kps)[2] + shoulder_center[2]) /2
        
        # human height is around 6~8 head high, take average 7
        pred_human_height = get_human_height(pp_kps)
        if pred_human_height!=0:
            head_height = pred_human_height / pred_human_height_head_height_ratio
            normal_shoulder_length = shoulder_height_ratio * pred_human_height
        head_middle_y = shoulder_center[1] - head_height * (1+neck_to_head_height_ratio)/2

        # calculate shoulder angle
        head_offset_vec = np.array([0,0,1]).reshape((-1,1))
        cos_ratio = shown_shoulder_length / normal_shoulder_length
        cos_ratio = np.clip(cos_ratio, -1, 1)
        shoulder_ang = np.arccos(cos_ratio)
        head_offset_vec = (y_rot_mat(shoulder_ang) @ head_offset_vec).reshape(-1)
        
        if not face_to_us(pp_kps):
            head_offset_vec *= -1
        head_cen_to_shoulder_cen = head_middle_y - shoulder_center[1]
        head_middle_coor = shoulder_center[:2] - head_cen_to_shoulder_cen * ortho_vec + head_offset_vec[:2] * normal_shoulder_length / 4
        
        pred_head_bbox_x1, pred_head_bbox_x2 = int(head_middle_coor[0] - head_width/2), int(head_middle_coor[0] + head_width/2)
        pred_head_bbox_y1, pred_head_bbox_y2 = int(head_middle_coor[1] - head_height/2), int(head_middle_coor[1] + head_height/2)
        box = ((pred_head_bbox_x1, pred_head_bbox_y1), (pred_head_bbox_x2, pred_head_bbox_y2))
        box_from_face = False
    
    else:
        box = None
        box_from_face = None
        conf = None
    return box, box_from_face, conf