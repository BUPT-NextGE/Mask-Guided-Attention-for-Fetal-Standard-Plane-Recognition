import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import MyDataset
from model.resnet_v1 import resnet50_v1
from model.resnet_v3 import resnet50_v3
from sklearn.metrics import classification_report

from model.resnet_v1_dual  import MyEnsemble
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2)

# We use pretrained torchvision models here
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

data_transform = {
    "train": transforms.Compose([transforms.Resize((224,224)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
    "val": transforms.Compose([transforms.Resize((224,224)),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}
modelA = resnet50_v3()
model_weight_path = "/mnt/DataDrive164/lirj/medical/medical_dataset/pre_weight/resnet50-pre.pth"
modelA.load_state_dict(torch.load(model_weight_path, map_location={'cuda:0': 'cuda:2'}))
modelB = resnet50_v3()
modelB.load_state_dict(torch.load(model_weight_path, map_location={'cuda:0': 'cuda:2'}))

net = MyEnsemble(modelA, modelB,nb_classes=2)
print(net)
net.to(device)
train_dataset = MyDataset.MyDataset('/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/10_cross/fold_1/origin/',
                                    '/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/10_cross/fold_0/filter/',
                                    data_transform["train"], 'train')

train_num = len(train_dataset)
validate_dataset = MyDataset.MyDataset('/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/10_cross/fold_1/origin/',
                                       '/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/10_cross/fold_0/filter/',
                                       data_transform["val"], 'val')

val_num = len(validate_dataset)

batch_size = 64
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=nw)

validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw)
print("using {} images for training, {} images for validation.".format(train_num,
                                                                       val_num))

loss_function = nn.CrossEntropyLoss()
# loss_function = FocalLoss(gama=2., size_average=True, weight=None)

params = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.Adam(net.parameters(), lr=5e-4)

epochs = 1
best_acc = 0.0
save_path = '/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/12_cross_model/2_grey/resnet_dual_erasev3_singlefold/5e-4_fold1'
os.makedirs(save_path, exist_ok=True)
for epoch in range(epochs):
    net.train()
    acc = 0.0
    running_loss = 0.0
    train_bar = tqdm(train_loader)
    for step, data in enumerate(train_bar):
        images1, images2, labels = data
        optimizer.zero_grad()
        logits = net(images1.to(device), images1.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        predict = torch.max(logits, dim=1)[1]
        acc += torch.eq(predict, labels.to(device)).sum().item()
        optimizer.step()
        running_loss += loss.item()

    train_accurate = acc / train_num
    predict_y_list=[]
    labels_list=[]

   # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for val_data in val_bar:
            images1, images2, labels = val_data
            optimizer.zero_grad()
            outputs = net(images1.to(device), images1.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, labels.to(device)).sum().item()
            predict_y_list.append(predict_y)
            labels_list.append(labels)

    val_accurate = acc / val_num

    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path+"/resNet50_%02d_%.3f.pth"%(epoch,val_accurate))

    print('[epoch %d] train_loss: %f train_accuracy: %f  val_accuracy: %f max_val_accuracy: %f' %
          (epoch + 1, running_loss / step,train_accurate, val_accurate,best_acc))
print('Finished Training')