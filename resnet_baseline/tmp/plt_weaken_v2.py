from model.resnet_weaken_v2 import resnet50_v3
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

weights_path = "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/12_cross_model/2_grey/resnet_weaken_v2/resnet50_0.0002_min0.3_seed20_fold3/resNet50_0_38_95.150.pth"

net.load_state_dict(torch.load(weights_path, map_location={'cuda:0': 'cuda:2'}))
save_path="/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/13_plt/weaken_v2_min0.3_fold0/"
os.makedirs(save_path, exist_ok=True)


net.to(device)


loader = transforms.Compose([

transforms.ToTensor()])

unloader = transforms.ToPILImage()
def gcam_to_mask(gcam, omega=100, sigma=0.5):
    mask = torch.sigmoid(omega * (gcam - sigma))
    return mask
# validate
net.eval()
with torch.no_grad():
    i=0
    for val_data in validate_loader:
        print("——————————————————————————————————————————————————————————")
        val_images, val_labels = val_data
        x,x_1,z,y_a,y_mask = net(val_images.to(device))
        print(x.shape)
        print(z.shape)
        print(y_a.shape)
        print(y_mask.shape)

        mask_sig=gcam_to_mask(y_mask)

        image = y_mask.cpu().clone()
        image_sig = mask_sig.cpu().clone()


        image = image.squeeze(0)
        image_sig=image_sig.squeeze(0)

        image = unloader(image)
        image_sig=unloader(image_sig)

        image_p=save_path+str(i)+'.jpg'
        image_p_sig=save_path+str(i)+'_sig.jpg'
        image.save(image_p)
        image_sig.save(image_p_sig)
        i+=1

