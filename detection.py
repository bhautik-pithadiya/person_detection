import torch
import time
import cv2
import torch.backends.cudnn as cudnn
from PIL import Image
import numpy as np
import os
import tqdm
import json

import super_gradients
from super_gradients.training import models
from super_gradients.common.object_names import Models

def loading_yolobnas():
    yolo_nas = models.get("yolo_nas_s", pretrained_weights="coco")
    return yolo_nas


def detection_in_img(img_path,yolo_nas):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = cv2.imread(img_path)

    # Get the height and width of the image
    height, width, _ = image.shape
    
    
    model_predictions  = list(yolo_nas.predict(img_path,conf = 0.5,)._images_prediction_lst)

    # model_pred = model_predictions[0].prediction()
    bboxes_xyxy = model_predictions[0].prediction.bboxes_xyxy.tolist()

    # bboxes_xyxy in json
    # print(model_pred)

    confidence = model_predictions[0].prediction.confidence.tolist()
    labels = model_predictions[0].prediction.labels.tolist()
    # lengths = [len(labels),len(confidence),len(bboxes_xyxy)] 
    labels = [int(label) for label in labels]

    person_bboxes_xyxy = [bbox for i, bbox in enumerate(bboxes_xyxy) if labels[i] == 0]
    person_confidence = [conf for i, conf in enumerate(confidence) if labels[i] == 0]
    person_labels = [label for label in labels if label == 0]

    # write a txt file
    output_filepath = img_path[:-4]
    
    with open(output_filepath + '.txt', 'w+') as f:
        for i, bbox in enumerate(person_bboxes_xyxy):
            x1,y1,x2,y2 = bbox

            # Normalizing coordinates
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width_ann = x2 - x1
            height_ann = y2 - y1
            center_x = center_x / width
            center_y = center_y / height
            width_ann = width_ann / width
            height_ann = height_ann / height
            f.write(str(person_labels[i]) + ' ' +  str(center_x) + ' ' + str(center_y) + ' ' + str(width_ann) + ' ' + str(height_ann) + '\n')


    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    folderpath = 'output_frames/'
    list_img = os.listdir(folderpath)
    print('*********************** Loading model...*******************************')
    model = loading_yolobnas()
    print('*********************** Model loaded!*******************************')
    # print(list_img)
    # with tqdm

    for img in tqdm.tqdm(list_img):
        img_path = folderpath + img
        detection_in_img(img_path,model)

    # testing for single image
    # path = folderpath + 'frame_1_48049.jpg'
    # detection_in_img(path,model)
