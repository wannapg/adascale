import config 
import numpy as np 
import os 
import pandas as pd 
import torch 

from PIL import Image, ImageFile 
from torch.utils.data import Dataset, DataLoader
from utils import (
    iou_width_height as iou, 
    non_max_suppression as nms, 
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset) : 
    def __init__(
        self,
        csv_file,
        img_dir, label_dir, 
        anchors, 
        image_size, 
        S = [13,26,52],
        C = 20, #classes
        transform = None,
    ): 
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = 'C:/workspace/darknet/data/MSCOCO/val2017/class/'
        self.label_dir = label_dir
        self.transform = transform
        self.S = S #grid size
        self.anchors = torch.tensor(anchors[0]+anchors[1]+anchors[2])
        self.num_anchors= self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors //3
        self.C = C
        self.ignore_iou_thresh = 0.5

def __len__(self): 
    return len(self.annotations)

def __getitem__(self, index, image_name): 
    label_path = 'C:/workspace/darknet/scripts/coco/labels/val2017'
    os.chdir(label_path)
    files = os.listdir(label_path)
    #ndmin? 
    #[class, x,y,w,h]
    bboxes = (np.loadtxt(fname=image_name, delimiter = " ", ndmin = 2)).toList()
    img_path = os.path.join(self.img_dir, image_name)
    image = np.array(Image.open(img_path).convert("RGB"))
    
    if self.transform : 
        augmentations = self.transform(image=image, bboxes = bboxes)
        image = augmentations["image"]
        bboxes = augmentations["bboxes"]

    #[probablility of obj[0 또는 1] ,x,y,w,h,class] total 6 values
    targets = [torch.zeros((self.num_anchors //3 , S, S, 6)) for s in self.S] 

    for box in bboxes : 
        iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
        anchor_indices = iou_anchors.argsort(descending=True , dim = 0 )
        class_label , x, y, width, height = box 
        has_anchor = [False , False, False] 

        for anchor_idx in anchor_indices : 
            scale_idx = anchor_idx // self.num_anchors_per_scale #0,1,2
            anchor_on_scale = anchor_idx %self.num_anchors_per_scale #0,1,2
            S = self.S[scale_idx]
            i, j = int(S*y), int(S*x) #which cell 
            anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

            if not anchor_taken and not has_anchor[scale_idx] : 
                targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                x_cell, y_cell = S*x - j , S*y - i #both are between [0,1]
                width_cell, height_cell = (
                    width * S, 
                    height * S
                )
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                targets[scale_idx][anchor_on_scale,i,j,1:5] = box_coordinates
                targets[scale_idx][anchor_on_scale, i, j, 1:5] = int(class_label)
            
            elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_threseh : 
                targets[scale_idx][anchor_on_scale, i, j, 0 ] = -1 #ignore this prediction

    return image, tuple(targets)