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

def generate_frame(index, name, rel_path, frame_label, image_size):
    frame = {}
    # name
    frame['name'] = name

    # url
    frame['url'] = os.path.join(rel_path, name)

    # videoName
    frame['videoName'] = ""
    # attributes
    # frame['attributes'] = 'null' 
    # timestamp
    # frameIndex
    frame['index'] = index
    # size
    # frame['size'] = {'height': image_size[0], 
    #                  'width': image_size[1]}

    # labels
    frame['labels'] = []
    # labels[key][id]{'xywh', 'conf', 'landmarks', 'class_num', 'xyxyconf'}
    for i in range(len(frame_label)):
        # frame['labels'].append(cur_annotation)
        frame['labels'].append(generate_2dlabel(i, frame_label[i]['xywh'], image_size[1], image_size[0]))



    return frame


def load_yolo(label_path, image_src):
    with open(label_path, 'r') as f:
        labels = json.load(f)
    
    output_labels = []# list of dicts
    # output_labels['frames'] = []
    # output_labels['config'] = []
    # # attr = {'name': ''}
    # output_labels['config'].append({'attributes': []})
    # output_labels['config'].append({'categories': ["face"]})


    # an image (1920, 1440)
    indx = 0
    for key in labels:
        # print(key)
        output_labels.append(generate_frame(indx, key, image_src, labels[key], (1440, 1920)))
        indx += 1
        print("processed: {}".format(key))


    # labels.key()
    return output_labels

def save_labels(output_labels, output_dir):

    with open(os.path.join(output_dir, "processed_labels.json"), "w") as output_file:
        json.dump(output_labels, output_file, indent=2)

if __name__=="__main__":
    processed_labels = load_yolo("/path/to/yolo_detections.json", "/items/images/camera6")
    save_labels(processed_labels, "/path/to/dest")
    # json.dumps(processed_labels, indent=2)
