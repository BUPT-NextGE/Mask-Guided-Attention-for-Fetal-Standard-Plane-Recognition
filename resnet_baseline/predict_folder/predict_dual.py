import os
import json

import torch
from PIL import Image

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import MyDataset
from model.resnet_v2 import resnet50_v2
from model.resnet_v1_dual  import MyEnsemble
import numpy as np
import random
import cv2
import time


# We use pretrained torchvision models here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform =transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


modelA = resnet50_v2(num_classes=2)
modelB = resnet50_v2(num_classes=2)

# create model
net = MyEnsemble(modelA, modelB, nb_classes=2).to(device)

# load model weights
weights_path = "/mnt/DataDrive164/lirj/medical/weights_backup/dual_erase/resNet50_2_68_95.602.pth"
net.load_state_dict(torch.load(weights_path, map_location=device))



predict_path = '/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/2_split/2_grey/val/nonstandard/'
time_start = time.time()
for pic_name in os.listdir(predict_path):
    print(pic_name)
    img_path = predict_path + pic_name
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    net.eval()
    with torch.no_grad():
        output=net(img.to(device),img.to(device)).cpu()
        predict=torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        print(predict_cla)
time_end = time.time()
time_result = str((round((time_end - time_start), 2))) + '秒'
print(time_result)