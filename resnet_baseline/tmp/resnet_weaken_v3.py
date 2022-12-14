import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model.resnet_weaken_v3 import resnet50_v3
from model.resnet_v1 import resnet50_v1

import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2)


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

print("using {} device.".format(device))

data_transform = {
    "train": transforms.Compose([transforms.Resize((224,224)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize((224,224)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = os.path.join(data_root, "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_cross/fold_0", "origin")  # flower data set path

assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transform["train"])
train_num = len(train_dataset)

flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 64
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

print('Using {} dataloader workers every process'.format(nw))

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=nw)

validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw)

print("using {} images for training, {} images for validation.".format(train_num,
                                                                       val_num))

net = resnet50_v3()
# load pretrain weights
model_weight_path = "/mnt/DataDrive164/lirj/medical/medical_dataset/pre_weight/resnet50-pre.pth"
assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

net.load_state_dict(torch.load(model_weight_path, map_location={'cuda:0': 'cuda:2'}))


num_fc_ftr = net.fc.in_features  # 获取到fc层的输入
net.fc = nn.Linear(num_fc_ftr, 2)  # 定义一个新的FC层

net.to(device)

# define loss function
loss_function = nn.CrossEntropyLoss()


params = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.Adam(net.parameters(), lr=0.00015)

epochs = 80
best_acc = 0.0
save_path = '/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/12_cross_model/2_grey/resnet_weaken_2/fold0_0.00015'

os.makedirs(save_path, exist_ok=True)
train_steps = len(train_loader)
for epoch in range(epochs):
    print("============================================================================================")

    # train
    net.train()
    acc=0.0
    running_loss = 0.0
    train_bar = (train_loader)
    for step, data in enumerate(train_bar):
        # print("——————————————————————————————————————————————————————————")
        images, labels = data
        optimizer.zero_grad()
        outputs,x_1,z,y_A,y_mask = net(images.to(device))
        # print(x_1.shape)
        # print(z.shape)
        # print(y_A.shape)
        # print(y_mask.shape)
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        predict = torch.max(outputs, dim=1)[1]
        acc += torch.eq(predict, labels.to(device)).sum().item()
        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}]  loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)


    train_accurate = acc / train_num




    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs,x_1,z,y_A,y_mask = net(val_images.to(device))
            # print(x_1.shape)
            # print(z.shape)
            # print(y_A.shape)
            # print(y_mask.shape)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                       epochs)

    val_accurate = acc / val_num


    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path+"/resNet50_filter_2_%.3f.pth"%val_accurate)
    print('[epoch %d] train_loss: %f train_accuracy: %f  val_accuracy: %f max_val_accuracy: %f' %
          (epoch + 1, running_loss / step,train_accurate, val_accurate,best_acc))

print('Finished Training')



