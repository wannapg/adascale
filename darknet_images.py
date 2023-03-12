import argparse
import os
import glob
import random
#from typing_extensions import Required
import darknet
import time
import cv2
import numpy as np
import darknet

from numpy import genfromtxt
import torch.nn as nn

from sklearn import datasets
import matplotlib.pyplot as plt 
from glob import glob
from sklearn.model_selection import train_test_split
import random 

#tensor
from PIL import Image
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


########################################################
#0) prepare data 
#1) Design model (input, output size, forward pass)
#2) loss and optimizer construct
#3) training loop 
#   - forward 
#   - backward 
#   - update weights
#########################################################

#training data generation
def data_gen(network, class_names, class_colors, thresh, args): 
    #training scale
    scales = [128,320,416,608]
    num_scales  = 4

    #image path, label path
    path = 'C:/workspace/darknet/data/MSCOCO/val2017/class'
    label_path = 'C:/workspace/darknet/scripts/coco/labels/val2017'
    os.chdir(path)
    files = os.listdir(path)
    label_files = os.listdir(label_path)
    

    #scale regressor train data filedir  
    f = open('C:/workspace/darknet/data/scale_train.txt','w')

    #image name : .jpg
    #label filename: .txt
    for image_name, label_name in zip(files, label_files) : 
        #rand = random.randrange(0,3)
        min_loss = 1000 #loss 
        min_loss_scale_index = 10 

        #scale 개수 만큼 detection 진행
        for i in range(num_scales):
            #bbox answer 결과 가져오기
            if i == 0 : 
                answer = darknet.get_image_info(label_path,label_name)
                os.chdir(path)
            
            image, detections = image_detection_scales(
                image_name, network, class_names, class_colors, args.thresh,scales[i],scales[i]
                )
            darknet.print_detections(detections, label_name)
            loss = darknet.calculate_loss(detections, answer,scales[i])

            if min_loss > loss : 
                min_loss_index = i 
                min_loss = loss

        best_scale = scales[min_loss_scale_index]
        f.write(str(best_scale))
        f.write('\n')
        print(best_scale)

    f. close()
 


# step 0 ) make data
def data_scalereg( network, class_names, class_colors, thresh, args):

    scales = [128,320,416,608]
    num_scales  = 4

    #image dir 
    path = 'C:/workspace/darknet/data/MSCOCO/val2017'
    os.chdir(path)
    files = os.listdir(path)

    #scale regressor train data filedir  
    f = open('C:/workspace/darknet/data/scale_train.txt','w')

    for image_name in files : 
        best_scale = -1 
        min_num_objects = -1 
        confidence_array = [] #image의 confidence array 
        for i in range(num_scales) : 
            confidence_array.append([])
            confidence_array_i = [] # i scale confidence array 
            detected_objects_num = -1 
            prev_time = time.time()
            image, detections = image_detection_scales(
                image_name, network, class_names, class_colors, args.thresh,scales[i],scales[i]
                )
            if args.save_labels:
                save_annotations(image_name, image, detections, class_names)

            # number of detected objects check for min 
            detected_objects_num = darknet.num_detected_objects(detections, args.ext_output)
            #min_num_objects update
            if i == 0 : 
                min_num_objects = detected_objects_num
            if detected_objects_num < min_num_objects: 
                min_num_objects = detected_objects_num
            #confidence value 2 dim array for different scales 
            confidence_array_i =darknet.object_confidence(detections, args.ext_output)
            confidence_array_i = list(map(float , confidence_array_i))
            for h in range(detected_objects_num): 
                confidence_array[i].append(confidence_array_i[h])
            #print(confidence_array_i) 
            darknet.print_detections(detections, args.ext_output)
            fps = int(1/(time.time() - prev_time))
            #print("FPS: {}".format(fps))

            #view open cv window 
            #if not args.dont_show:
            #    cv2.imshow('Inference', image)
            #    if cv2.waitKey() & 0xFF == ord('q'):
            #        break

        #min_objects_detected loss calculation
        max_accuracy = -1 
        best_scale = scales[num_scales-1]

        darknet.calculate_loss(detections, image_name,scales[i],min_num_objects, args.ext_output)
        os.chdir(path)
        '''
        for m in range(num_scales - 1): 
            average = 0.0
            sum = 0.0
            for n in range(min_num_objects - 1) :   
                sum = sum +  confidence_array[m][n]
            if min_num_objects != 0 : 
                average =  sum / min_num_objects
            if max_accuracy < average : 
                #update
                min_index = i 
                best_scale = scales[num_scales-i-1]
                #save best scale data
        '''
        f.write(str(best_scale))
        f.write('\n')
        print(best_scale)

    f. close()
 
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

def train_scalereg(network):

        #0) prepare data
        #image dir 
        #data_dir = 'C:/workspace/darknet/data/MSCOCO/val2017'
        f = open('C:/workspace/darknet/data/scale_train.txt','r')

        path = 'C:/workspace/darknet/data/MSCOCO/val2017/class'
        os.chdir(path)
        files = os.listdir(path)


        print("transforming image data to tensor...")
        
        
        i = 0 
        for image_name in files : 
            img = Image.open(image_name)
            #img= img.resize((256,256))
            #np.array(image_name).shape
            data = np.array(img)
            imgToTensorTransformer = transforms.ToTensor() 
            tensorFromImg = imgToTensorTransformer(data)
            
        #tensorFromImg.reshape(256,256,4999)
        #trainset = trainset.torch.tensor()
        #x_train = torch.numpy()
        #print(x_train)
        #x_train = torch.FloatTensor(x_train)
        y_train = genfromtxt('C:/workspace/darknet/data/scale_train.txt', delimiter ='\n')#scale
        y_train = torch.tensor(y_train).float()
        #y_train = y_train.view(y_train.shape[0],1)

        #1) model
        input_size = 640
        output_size = 1 
        model = nn.Linear(input_size, output_size)

        #2) loss and optimizer 
        learning_rate = 0.001 
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

        #3) training loop 
        num_epochs = 2
        #batch_size = 512
        print("training model...")
        for epoch in range(num_epochs): 
            #forward pass and loss 
            y_predicted = model(tensorFromImg)
            y_predicted = y_predicted.float()
            loss = criterion(y_predicted, y_train)

            #backward pass , gradient calculation
            loss.backward()

            #update weights 
            optimizer.step()

            #empty gradients
            optimizer.zero_grad()

            #if (epoch+1)%10 == 0 : 
            print(f'epoch: {epoch+1} , loss = {loss.item():.4f}') 

        #save the whole model
        #saved model or
        FILE = "model.pth"
        torch.save(model,FILE)
        return model


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="yolov3.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default="./cfg/yolov3.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))


def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)


def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

#adascale regressor width, height
def image_detection_scales(image_path, network, class_names, class_colors, thresh, width, height):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    #width = darknet.network_width(network)
    #height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])

#detection with input scales
def image_classification_scales(image, network, class_name, width, height): 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = os.path.splitext(name)[0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def batch_detection_example():
    args = parser()
    check_arguments_errors(args)
    batch_size = 3
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )
    image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    images = [cv2.imread(image) for image in image_names]
    images, detections,  = batch_detection(network, images, class_names,
                                           class_colors, batch_size=batch_size)
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)
    print(detections)



def main():
    args = parser()
    check_arguments_errors(args)

    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )
    # train data generation
    #data_scalereg( network, class_names, class_colors, args.thresh, args)
    #data_gen( network, class_names, class_colors, args.thresh, args)


    #train scale regressor
    #train_scalereg(network)

    model= torch.load("model.pth")
    model.eval()

    path = 'C:/workspace/darknet/data/MSCOCO/val2017/class'
    os.chdir(path)
    files = os.listdir(path)

    for file in files: 
        img = Image.open(file)
        data = np.array(img)
        imgToTensorTransformer = transforms.ToTensor() 
        tensorFromImg = imgToTensorTransformer(data)
        model = nn.AdaptiveAvgPool2d(1)

        scale = model(torch.tensor(tensorFromImg))
        print("scale: ", scale)


'''
    images = load_images(args.input)


    index = 0
    while True:
        # loop asking for new image paths if no list is given
        if args.input:
            if index >= len(images):
                break
            image_name = images[index]
        else:
            image_name = input("Enter Image Path: ")
        prev_time = time.time()
        image, detections = image_detection(
            image_name, network, class_names, class_colors, args.thresh
            )
        if args.save_labels:
            save_annotations(image_name, image, detections, class_names)
        darknet.print_detections(detections, args.ext_output)
        fps = int(1/(time.time() - prev_time))
        print("FPS: {}".format(fps))
        if not args.dont_show:
            cv2.imshow('Inference', image)
            if cv2.waitKey() & 0xFF == ord('q'):
                break
        index += 1
        
'''

if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    main()