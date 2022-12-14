import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model.model_dual import resnet50
import cv2

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # class_indict = json.load(json_file)

    # create model
    model = resnet50(num_classes=2).to(device)

    # load model weights
    weights_path = "/home/ubuntu/users/lirj/medical_image_classfication/attention_crop/resnet_baseline/resnet/resnet50_dual/resNet50_0.720.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    predict_path='/mnt/DataDrive164/lirj/medical/medical_dataset/resnet_baseline/dataset/origin/val/nonstandard/'

    predict_mask_path='/mnt/DataDrive164/lirj/medical/medical_dataset/resnet_baseline/dataset/filter/val/nonstandard/'

    # save_error_path="/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/8_error/standard"
    # os.makedirs(save_error_path, exist_ok=True)
    for pic_name in os.listdir(predict_path):
        # load image
        img_path =predict_path+pic_name
        img_mask_path=predict_mask_path+pic_name
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        img = img.convert('RGB')
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)



        img_mask = Image.open(img_mask_path)
        img_mask = img_mask.convert('RGB')
        # [N, C, H, W]
        img_mask = data_transform(img_mask)
        # expand batch dimension
        img_mask = torch.unsqueeze(img_mask, dim=0)



        # prediction
        model.eval()
        with torch.no_grad():
            output = torch.squeeze(model(img.to(device),img_mask.to(device))).cpu()

            # logits = net(images1.to(device), images2.to(device))


            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "{} prob: {:.3}".format(pic_name,predict[predict_cla].numpy())
        print(print_res,class_indict[str(predict_cla)])


    print(error)


if __name__ == '__main__':
    main()
