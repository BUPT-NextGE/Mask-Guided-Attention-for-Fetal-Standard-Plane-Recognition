import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import MyDataset
from model_dual import resnet50

import numpy as np
import random
def main():
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(20)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    train_dataset = MyDataset.MyDataset('/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_dual/origin/',
                                        '/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_dual/filter/', data_transform["train"], 'train')

    train_num = len(train_dataset)
    validate_dataset = MyDataset.MyDataset('/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_dual/origin/',
                                           '/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_dual/filter/', data_transform["val"], 'val')

    val_num = len(validate_dataset)


    batch_size = 16
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


    net = resnet50()
    net.load_state_dict(torch.load( "/mnt/DataDrive164/lirj/medical/medical_dataset/pre_weight/resnet50-pre.pth"), strict=False)



    num_fc_ftr = net.fc_dual.in_features  # 获取到fc层的输入
    net.fc_dual = nn.Linear(num_fc_ftr, 2)  # 定义一个新的FC层
    net.to(device)
    loss_function = nn.CrossEntropyLoss()

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(net.parameters(), lr=10)

    epochs = 100
    best_acc = 0.0
    save_path = './resnet/resnet50_dual'
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        # train
        net.train()
        acc = 0.0

        running_loss = 0.0
        train_bar = tqdm(train_loader)

        # for step, data in enumerate(train_bar):
        #     images1, images2, labels = data
        #     optimizer.zero_grad()
        #     outputs = net(images1.to(device), images2.to(device))
        #     loss = loss_function(outputs, labels.to(device))
        #     loss.backward()
        #     optimizer.step()

        for step, data in enumerate(train_bar):
            images1, images2, labels = data
            optimizer.zero_grad()
            logits = net(images1.to(device), images1.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            predict = torch.max(logits, dim=1)[1]
            # print('predict:',predict)
            acc += torch.eq(predict, labels.to(device)).sum().item()
            optimizer.step()
            running_loss += loss.item()


        train_accurate = acc / train_num
        print("训练集准确率：",train_accurate)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                images1, images2, labels = val_data
                optimizer.zero_grad()
                outputs = net(images1.to(device), images1.to(device))

                # logits = net(images1.to(device), images2.to(device))
                # predict = torch.max(logits, dim=1)[1]
                # acc += torch.eq(predict, labels.to(device)).sum().item()





                predict_y = torch.max(outputs, dim=1)[1]

                acc += torch.eq(predict_y, labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %f  val_accuracy: %f' %
              (epoch + 1, running_loss / step, val_accurate))
        # print(val_accurate)
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path+"/resNet50_%.3f.pth"%val_accurate)

    print('Finished Training')


if __name__ == '__main__':
    main()
