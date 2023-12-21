import torch
import time
import cv2
import torch.backends.cudnn as cudnn
from PIL import Image
import numpy as np
import os
import tqdm

import super_gradients
from super_gradients.training import models
from super_gradients.common.object_names import Models

def loading_yolobnas():
    yolo_nas = models.get("yolo_nas_s", pretrained_weights="coco")
    return yolo_nas


def detection_in_img(img_path,yolo_nas):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_predictions  = list(yolo_nas.predict(img_path,conf = 0.5,)._images_prediction_lst)

    bboxes_xyxy = model_predictions[0].prediction.bboxes_xyxy.tolist()
    # confidence = model_predictions[0].prediction.confidence.tolist()
    labels = model_predictions[0].prediction.labels.tolist()
    # lengths = [len(labels),len(confidence),len(bboxes_xyxy)] 
    labels = [int(label) for label in labels]

    person_bboxes_xyxy = [bbox for i, bbox in enumerate(bboxes_xyxy) if labels[i] == 0]
    # person_confidence = [conf for i, conf in enumerate(confidence) if labels[i] == 0]
    person_labels = [label for label in labels if label == 0]

    # write a txt file
    output_filepath = img_path[:-4] 
    with open(output_filepath + '.txt', 'w') as f:
        for i, bbox in enumerate(person_bboxes_xyxy):
            f.write('0 ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + '\n')


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
    # for img in tqdm.tqdm(list_img):
    #     img_path = folderpath + img
    #     detection_in_img(img_path,model)

    detection_in_img*(folderpath + list_img[0],model)
