from model.resnet_weaken_v1 import resnet50_v3
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import numpy as np
import random
from PIL import Image
import cv2
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_transform = {

    "val": transforms.Compose([transforms.Resize((224,224)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}



data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = os.path.join(data_root, "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_cross/fold_0", "origin")  # flower data set path




batch_size = 1
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers



validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw)


net = resnet50_v3(num_classes=2)

weights_path = "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/3_model/2_grey/resnet/resnet50_org/resNet50_filter_2_0.928.pth"

net.load_state_dict(torch.load(weights_path, map_location={'cuda:0': 'cuda:2'}))


net.to(device)


loader = transforms.Compose([

transforms.ToTensor()])

unloader = transforms.ToPILImage()
# validate
net.eval()
with torch.no_grad():
    i=0
    for val_data in validate_loader:
        print("——————————————————————————————————————————————————————————")
        val_images, val_labels = val_data
        x,z,A,mask = net(val_images.to(device))
        print(x.shape)
        print(z.shape)
        print(A.shape)
        print(mask.shape)

        image = mask.cpu().clone()

        image = image.squeeze(0)

        image = unloader(image)
        image_p="./img/"+str(i)+'.jpg'
        image.save(image_p)
        i+=1

