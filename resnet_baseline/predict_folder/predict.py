import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model.resnet_v2 import resnet50_v2
import cv2
import time
def main():
    time_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    error=0.0
    # read class_indict
    json_path = '../class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    # class_indict = json.load(json_file)

    # create model
    model = resnet50_v2(num_classes=2).to(device)

    # load model weights
    weights_path = "/mnt/DataDrive164/lirj/medical/weights_backup/erase/resNet50_1_53_93.533.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    predict_path='/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/2_split/2_grey/val/standard/'
    save_error_path="/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/8_error/standard"
    os.makedirs(save_error_path, exist_ok=True)
    y_true, y_pred = [], []
    for pic_name in os.listdir(predict_path):
        # load image
        img_path =predict_path+pic_name
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        img = img.convert('RGB')
        img_copy=cv2.imread(img_path)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)

        # prediction
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        print_res = "{} prob: {:.3}".format(pic_name,
                                                     predict[predict_cla].numpy())
        print(print_res,class_indict[str(predict_cla)])
    time_end = time.time()
    time_result = str((round((time_end - time_start), 2))) + '???'
    print(time_result)




if __name__ == '__main__':
    main()
