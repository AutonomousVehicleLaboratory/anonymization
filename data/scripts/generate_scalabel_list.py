import os
import json

def generate_2dlabel(id, xywh, image_width, image_height):
    x1 = (xywh[0] - xywh[2] / 2) * image_width
    y1 = (xywh[1] + xywh[3] / 2) * image_height
    x2 = (xywh[0] + xywh[2] / 2) * image_width
    y2 = (xywh[1] - xywh[3] / 2) * image_height

    label = {}
    # id
    label['id'] = id
    # index
    # category
    label['category'] = "face"
    # manualShape
    # manualAttributes
    # score
    # attributes
    label['attributes'] = {}
    # box2d
    bb = {"x1": x1,
             "x2": x2, 
             "y1": y1,
             "y2": y2}

    label['box2d'] = bb

    # box3d
    # poly2d
    # rle
    # graph
    

    return label

def generate_frame(index, name, rel_path):
    frame = {}

    frame["index"] = index
    # name
    frame['name'] = name

    # url
    frame['url'] = os.path.join(rel_path, name)

    # videoName
    frame['videoName'] = "0"

    return frame


def load_image_list(image_dir, image_src):
    name_list = sorted(os.listdir(image_dir))
    
    image_list = []# list of dicts

    for indx, key in enumerate(name_list):
        image_list.append(generate_frame(indx, key, image_src))
        print("processed: {}".format(key))

    # labels.key()
    return image_list

def save_labels(output_labels, output_dir):

    with open(os.path.join(output_dir, "image_list.json"), "w") as output_file:
        json.dump(output_labels, output_file, indent=2)

if __name__=="__main__":
    image_dir = "/home/henrymelodic/Documents/data/avl/2021-10-26-17-03-55/avt_cameras_camera1_image_color_compressed/"
    image_src = "http://localhost:8686/items/2021-10-26-17-03-55/avt_cameras_camera1_image_color_compressed/"
    dest_dir = "../../output_dir/"
    image_list = load_image_list(image_dir, image_src)
    save_labels(image_list, dest_dir)
    # json.dumps(processed_labels, indent=2)
