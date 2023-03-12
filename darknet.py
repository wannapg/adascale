#!/usr/bin/env python3

"""
Python 3 wrapper for identifying objects in images

Running the script requires opencv-python to be installed (`pip install opencv-python`)
Directly viewing or returning bounding-boxed images requires scikit-image to be installed (`pip install scikit-image`)
Use pip3 instead of pip on some systems to be sure to install modules for python3
"""

from ctypes import *
import math
import random
import os
import numpy as np

from typing import ForwardRef
import torch 
import torch.nn as nn 

from utils import (
    cells_to_bboxes,
    intersection_over_union,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("best_class_idx", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]

class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


def network_width(net):
    return lib.network_width(net)


def network_height(net):
    return lib.network_height(net)
    
def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax



def class_colors(names):
    """x
    Create a dict with one random BGR color for each
    class name
    """
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}


def load_network(config_file, data_file, weights, batch_size=1):
    """
    load model description and weights from config files
    args:
        config_file (str): path to .cfg model file
        data_file (str): path to .data model file
        weights (str): path to weights
    returns:
        network: trained model
        class_names
        class_colors
    """
    network = load_net_custom(
        config_file.encode("ascii"),
        weights.encode("ascii"), 0, batch_size)
    metadata = load_meta(data_file.encode("ascii"))
    class_names = [metadata.names[i].decode("ascii") for i in range(metadata.classes)]
    colors = class_colors(class_names)
    return network, class_names, colors

#class, bbox answer info
def get_image_info(label_path, label_name): 
    os.chdir(label_path)
    info = (np.loadtxt(fname=label_name, delimiter = " ", ndmin = 2)).tolist()
    return info


def print_detections(detections, coordinates=False):
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        if coordinates:
            print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
        else:
            print("{}: {}%".format(label, confidence))
        print(label,x,y,w,h)


def print_detections_txt(image_name, detections, coordinates = False) : 
    #txt_path = 'C:/Users/wanna/OneDrive/onedrive_workspace/denoising/data/bdd100k/images/100k/train_noise_level/2_320x180_dncnn_color_blind_txt'
    txt_path = 'C:/Users/wanna/OneDrive/onedrive_workspace/denoising/DnCNN/KAIR/results/2_fdncnn_color_txt_5'
    
    os.chdir(txt_path)
    files = os.listdir(txt_path)
    image_name =image_name.split('.')
    f = open(image_name[0]+'.txt', 'w')

    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        if label=='pedestrian': 
                category_id  = 0 
        elif  label== 'rider': 
            category_id = 1
        elif  label== 'car': 
            category_id = 2
        elif  label== 'truck': 
            category_id = 3
        elif  label== 'bus': 
            category_id = 4
        elif  label== 'train': 
            category_id = 5
        elif  label== 'motorcycle': 
            category_id = 6
        elif  label== 'bicycle': 
            category_id = 7
        elif  label== 'trafficlight': 
            category_id = 8
        elif  label== 'trafficsign': 
            category_id = 9 
        else : 
            category_id = -1

        x1,y1,x2,y2= bbox2points(bbox)

        f.write(label)
        f.write(" ")
        f.write(confidence)
        f.write(" ")
        if x-w/2 > 0: 
            f.write(str(x1))
        else :
            f.write(str(0))
        f.write(" ")
        f.write(str(y1))
        f.write(" ")
        f.write(str(x2))
        f.write(" ")
        if y-h/2 > 0: 
            f.write(str(y2)) 
        else : 
            f.write(str(0))
        f.write("\n")

        #print(label, confidence,str(x-w/2),str(x+h/2),str(x+w/2),str(y-h/2))
    f.close()

    
#scale regressor training위한 confidence array return 하기 
def object_confidence(detections, coordinates = False): 
    conf_arr = [] 
    for label, confidence , bbox in detections : 
        conf_arr.append(confidence)
    conf_arr.reverse()
    return conf_arr


def calculate_loss(detections, answer, img_scale): 

    loss = 0 
    # 0 : class, 1: x, 2: y, 3: w, 4: h 
    for label, confidence, bbox in detections:
        answer_taken = []
        max_detection_iou = 0 
        #most fit answer with iou
        for lines in answer: 
            iou_detection = iou(torch.tensor(bbox),torch.tensor([lines[1]*img_scale,lines[2]*img_scale,lines[3]*img_scale,lines[4]*img_scale]))
            if max_detection_iou <= iou_detection : 
                max_detection_iou = iou_detection
                answer_taken = [lines[0],lines[1]*img_scale,lines[2]*img_scale,lines[3]*img_scale,lines[4]*img_scale]

        #object loss 
        #box_detection = torch.cat([nn.Sigmoid(bbox[...,1:3]), torch.exp[...,3:5]],dim = -1)
        ious = intersection_over_union(torch.tensor(bbox), torch.tensor(answer_taken))
        object_loss = nn.BCEWithLogitsLoss(weight=None, reduce=False)

        #print(object_loss)

        #box coordinate loss (localization loss)
        #box_loss = nn.MSELoss(bbox,answer_taken[1:4])

        #class loss 
        #class_loss = nn.CrossEntropyLoss()

        #loss = loss + object_loss + box_loss +class_loss()

    return object_loss


#return detected number of objects 
def num_detected_objects(detections, coordinates = False ): 
    #confidence 값이 50% 이상되는 값의 개수만을 count 하여 object로 추출 
    count = 0 
    for label,confidence,bbox in detections: 
        if float(confidence)  >= 50: 
            count = count + 1 
    return count 

def draw_boxes(detections, image, colors):
    import cv2
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image


def decode_detection(detections):
    decoded = []
    for label, confidence, bbox in detections:
        confidence = str(round(confidence * 100, 2))
        decoded.append((str(label), confidence, bbox))
    return decoded

# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# Malisiewicz et al.
def non_max_suppression_fast(detections, overlap_thresh):
    boxes = []
    for detection in detections:
        _, _, _, (x, y, w, h) = detection
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes.append(np.array([x1, y1, x2, y2]))
    boxes_array = np.array(boxes)

    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes_array[:, 0]
    y1 = boxes_array[:, 1]
    x2 = boxes_array[:, 2]
    y2 = boxes_array[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))
        # return only the bounding boxes that were picked using the
        # integer data type
    return [detections[i] for i in pick]

def remove_negatives(detections, class_names, num):
    """
    Remove all classes with 0% confidence within the detection
    """
    predictions = []
    for j in range(num):
        for idx, name in enumerate(class_names):
            if detections[j].prob[idx] > 0:
                bbox = detections[j].bbox
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                predictions.append((name, detections[j].prob[idx], (bbox)))
    return predictions 


def remove_negatives_faster(detections, class_names, num):
    """
    Faster version of remove_negatives (very useful when using yolo9000)
    """
    predictions = []
    for j in range(num):
        if detections[j].best_class_idx == -1:
            continue
        name = class_names[detections[j].best_class_idx]
        bbox = detections[j].bbox
        bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
        predictions.append((name, detections[j].prob[detections[j].best_class_idx], bbox))
    return predictions


def detect_image(network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45):
    """
        Returns a list with highest confidence class and their bbox
    """
    pnum = pointer(c_int(0))
    predict_image(network, image)
    detections = get_network_boxes(network, image.w, image.h,
                                   thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(detections, num, len(class_names), nms)
    predictions = remove_negatives(detections, class_names, num)
    predictions = decode_detection(predictions)
    free_detections(detections, num)
    return sorted(predictions, key=lambda x: x[1])

def detect_image_adascale(network, class_names, image,scale, thresh=.5, hier_thresh=.5, nms=.45):
    """
        Returns a list with highest confidence class and their bbox
    """
    pnum = pointer(c_int(0))
    predict_image(network, image)
    detections = get_network_boxes(network, scale, scale, 
                                   thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(detections, num, len(class_names), nms)
    predictions = remove_negatives(detections, class_names, num)
    predictions = decode_detection(predictions)
    free_detections(detections, num)
    return sorted(predictions, key=lambda x: x[1])


if os.name == "posix":
    cwd = os.path.dirname(__file__)
    lib = CDLL(cwd + "/libdarknet.so", RTLD_GLOBAL)
elif os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    lib = CDLL("darknet.dll", RTLD_GLOBAL)
else:
    print("Unsupported OS")
    exit

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_batch_detections = lib.free_batch_detections
free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

free_network_ptr = lib.free_network_ptr
free_network_ptr.argtypes = [c_void_p]
free_network_ptr.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                   c_float, c_float, POINTER(c_int), c_int, c_int]
network_predict_batch.restype = POINTER(DETNUMPAIR)
