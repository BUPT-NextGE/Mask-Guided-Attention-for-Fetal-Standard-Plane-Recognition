import os
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import torch
import cv2
from PIL import Image
import glob


class MyDataset(Dataset):
    def __init__(self, origin_root, filter_root, transform):
        self.origin_root = origin_root
        self.filter_root = filter_root
        self.images = []
        self.transform = transform
        self.labels = {}
        self.tmp=os.path.join(self.origin_root)
        self.img_list=[]
        self.path_list=[]
        images_path = glob.glob(self.tmp+'/*/*.bmp')
        for index_path in images_path:
            folder_path, file_name = os.path.split(index_path)
            folders = folder_path.split('/')
            special_path = folders[-1]+'/'+file_name
            self.images.append(special_path)
            label = 0 if folders[-1] == 'nonstandard' else 1
            self.labels[special_path] = label
            self.img_list.append(file_name)
            self.path_list.append(index_path)




    def __getitem__(self, index):
        img_name = self.images[index]
        img_path1 = os.path.join(self.origin_root,  img_name)
        img_path2 = os.path.join(self.filter_root,  img_name)
        pil_image1 = Image.open(img_path1).convert("RGB")
        pil_image2 = Image.open(img_path2).convert("RGB")


        if self.transform:
            data1 = self.transform(pil_image1)
            data2 = self.transform(pil_image2)
        else:
            data1 = torch.from_numpy(pil_image1)
            data2 = torch.from_numpy(pil_image2)


        label = self.labels[img_name]

        return data1, data2, label

    def img(self):

        return self.img_list
    def path(self):

        return self.path_list

    def __len__(self):
        return len(self.images)
