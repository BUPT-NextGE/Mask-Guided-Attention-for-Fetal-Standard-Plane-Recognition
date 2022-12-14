import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import MyDataset
from model import resnet50
from model_dual_1 import MyEnsemble
import cv2

def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    error=0.0
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    modelA = resnet50()
    modelB = resnet50()

    # create model
    model = MyEnsemble(modelA, modelB, nb_classes=2)
    model.to(device)

    # load model weights
    weights_path = "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_dual/model/11/resNet50_0.949.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    predict_path='/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_dual/origin/val/standard/'

    predict_mask_path='/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_dual/filter/val/standard/'

    for pic_name in os.listdir(predict_path):
        # load image
        img_path =predict_path+pic_name
        img_mask_path=predict_mask_path+pic_name
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)



        img_mask = Image.open(img_mask_path)
        img_mask = img_mask.convert('RGB')
        img_mask = data_transform(img_mask)
        img_mask = torch.unsqueeze(img_mask, dim=0)

        # prediction
        model.eval()
        with torch.no_grad():
            output = model(img.to(device),img_mask.to(device))
            ps = torch.exp(output)
            topk, topclass = ps.topk(1, dim=1)
            print(topclass)
            # _,indices=torch.sort(output,descending=True)
            # print(indices)



    print(error)


if __name__ == '__main__':
    main()
