import os
import torch
import cv2
import time
import numpy as np

from PIL import Image
from sklearn.model_selection import KFold
from torch import nn
from torch import optim
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms
from pytorchtools import EarlyStopping
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2)


dataset_version = 'original' # original or augmented
img_shape = (224,224)
img_size = str(img_shape[0])+"x"+str(img_shape[1])

# Root directory of dataset
data_dir = "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/1_data/2_grey"

train_batch_size = 32
val_test_batch_size = 32
feature_extract = False
pretrained = True
h_epochs = 80
kfolds = 3

# Define transforms for input data
training_transforms = transforms.Compose([transforms.Resize((224,224), Image.LANCZOS),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])



total_set = datasets.ImageFolder(data_dir, transform=training_transforms)

# Defining folds
splits = KFold(n_splits = kfolds, shuffle = True, random_state = 42)
train_labels = {value: key for (key, value) in total_set.class_to_idx.items()}


save_path = '/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/12_cross_model/2_grey/resnet_erase/resnet50_2e-5_fold3'
os.makedirs(save_path, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def load_model():
    model = resnet50()
    model_weight_path = "/mnt/DataDrive164/lirj/medical/medical_dataset/pre_weight/resnet50-pre.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location={'cuda:0': 'cuda:2'}))
    num_fc_ftr = model.fc.in_features  # ?????????fc????????????
    model.fc = nn.Linear(num_fc_ftr, 2)  # ??????????????????FC???
    return model


def create_optimizer(model):


    if feature_extract:
        params_to_update = []
        for param in model.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

    else:
        n_params = 0
        for param in model.parameters():
            if param.requires_grad == True:
                n_params += 1

    # Loss function and gradient descent

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    return criterion.to(device), model.to(device), optimizer

print(total_set.imgs)
for fold, (train_idx, valid_idx) in enumerate(splits.split(total_set)):


    print('Fold : {}'.format(fold))

    # Train and val samplers
    train_sampler = SubsetRandomSampler(train_idx)
    print("Samples in training:", len(train_sampler))
    valid_sampler = SubsetRandomSampler(valid_idx)
    print("Samples in test:", len(valid_sampler))


    # Train and val loaders
    train_loader = torch.utils.data.DataLoader(
        total_set, batch_size=train_batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        total_set, batch_size=val_test_batch_size, sampler=valid_sampler)

    print(train_idx)
    for i in range(len(train_idx)):
        img_num=train_idx[i]
        # print(img_num)
        # print(total_set.imgs[img_num])
        img_name=total_set.imgs[img_num][0].split('/')[-1]
        print(img_name)
        img_label=total_set.imgs[img_num][0].split('/')[-2]
        print(img_label)
        if(fold==0):
            if(img_label=="standard"):
                img_tmp=cv2.imread(total_set.imgs[img_num][0])
                save_new_path="/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_cross/fold_0/train/standard/"
                os.makedirs(save_new_path, exist_ok=True)

                cv2.imwrite(save_new_path+img_name,img_tmp)
            else:
                img_tmp=cv2.imread(total_set.imgs[img_num][0])
                save_new_path="/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_cross/fold_0/train/nonstandard/"
                os.makedirs(save_new_path, exist_ok=True)

                cv2.imwrite(save_new_path+img_name,img_tmp)
        if (fold == 1):
            if (img_label == "standard"):
                img_tmp = cv2.imread(total_set.imgs[img_num][0])
                save_new_path = "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_cross/fold_1/train/standard/"
                os.makedirs(save_new_path, exist_ok=True)

                cv2.imwrite(save_new_path + img_name, img_tmp)
            else:
                img_tmp = cv2.imread(total_set.imgs[img_num][0])
                save_new_path = "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_cross/fold_1/train/nonstandard/"
                os.makedirs(save_new_path, exist_ok=True)

                cv2.imwrite(save_new_path + img_name, img_tmp)
        if (fold == 2):
            if (img_label == "standard"):
                img_tmp = cv2.imread(total_set.imgs[img_num][0])
                save_new_path = "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_cross/fold_2/train/standard/"
                os.makedirs(save_new_path, exist_ok=True)

                cv2.imwrite(save_new_path + img_name, img_tmp)
            else:
                img_tmp = cv2.imread(total_set.imgs[img_num][0])
                save_new_path = "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_cross/fold_2/train/nonstandard/"
                os.makedirs(save_new_path, exist_ok=True)

                cv2.imwrite(save_new_path + img_name, img_tmp)

    print(valid_idx)
    for i in range(len(valid_idx)):
        img_num = valid_idx[i]
        # print(img_num)
        # print(total_set.imgs[img_num])
        img_name = total_set.imgs[img_num][0].split('/')[-1]
        print(img_name)
        img_label = total_set.imgs[img_num][0].split('/')[-2]
        print(img_label)
        if (fold == 0):
            if (img_label == "standard"):
                img_tmp = cv2.imread(total_set.imgs[img_num][0])
                save_new_path = "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_cross/fold_0/val/standard/"
                os.makedirs(save_new_path, exist_ok=True)

                cv2.imwrite(save_new_path + img_name, img_tmp)
            else:
                img_tmp = cv2.imread(total_set.imgs[img_num][0])
                save_new_path = "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_cross/fold_0/val/nonstandard/"
                os.makedirs(save_new_path, exist_ok=True)

                cv2.imwrite(save_new_path + img_name, img_tmp)
        if (fold == 1):
            if (img_label == "standard"):
                img_tmp = cv2.imread(total_set.imgs[img_num][0])
                save_new_path = "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_cross/fold_1/val/standard/"
                os.makedirs(save_new_path, exist_ok=True)

                cv2.imwrite(save_new_path + img_name, img_tmp)
            else:
                img_tmp = cv2.imread(total_set.imgs[img_num][0])
                save_new_path = "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_cross/fold_1/val/nonstandard/"
                os.makedirs(save_new_path, exist_ok=True)

                cv2.imwrite(save_new_path + img_name, img_tmp)
        if (fold == 2):
            if (img_label == "standard"):
                img_tmp = cv2.imread(total_set.imgs[img_num][0])
                save_new_path = "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_cross/fold_2/val/standard/"
                os.makedirs(save_new_path, exist_ok=True)

                cv2.imwrite(save_new_path + img_name, img_tmp)
            else:
                img_tmp = cv2.imread(total_set.imgs[img_num][0])
                save_new_path = "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_cross/fold_2/val/nonstandard/"
                os.makedirs(save_new_path, exist_ok=True)

                cv2.imwrite(save_new_path + img_name, img_tmp)



