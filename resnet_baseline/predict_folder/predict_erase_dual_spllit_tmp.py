import os
import torch

import time
import numpy as np

from PIL import Image
from sklearn.model_selection import KFold
from torch import nn
from torch import optim
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms
from model.resnet_v2 import resnet50_v2
from model.resnet_v1 import resnet50_v1
from model.resnet_v1_dual  import MyEnsemble
from sklearn.metrics import classification_report

from pytorchtools import EarlyStopping
import random
import MyDataset
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)


dataset_version = 'original' # original or augmented
img_shape = (224,224)
img_size = str(img_shape[0])+"x"+str(img_shape[1])

# Root directory of dataset
data_dir = "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/1_data/2_grey"

train_batch_size = 64
val_test_batch_size = 64
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



# total_set = datasets.ImageFolder(data_dir, transform=training_transforms)
total_set=MyDataset.MyDataset('/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/1_data/2_grey',
                              '/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/1_data/mask',
                              training_transforms)

splits = KFold(n_splits = kfolds, shuffle = True, random_state = 42)


save_path = '/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/12_cross_model/2_grey/dual_erase_seed20_0.00020'
os.makedirs(save_path, exist_ok=True)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



def load_model():
    modelA = resnet50_v2()
    model_weight_path = "/mnt/DataDrive164/lirj/medical/weights_backup/dual_erase/resNet50_1_75_95.150.pth"
    modelB = resnet50_v2()
    model = MyEnsemble(modelA, modelB, nb_classes=2)
    model.load_state_dict(torch.load(model_weight_path, map_location={'cuda:0': 'cuda:2'}))

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

    optimizer = optim.Adam(model.parameters(), lr=0.00020)

    return criterion.to(device), model.to(device), optimizer

# Variables to store fold scores
train_acc = []
test_top1_acc = []
test_top5_acc = []
test_top5_acc = []
test_precision = []
test_recall = []
test_f1 = []
times = []

for fold, (train_idx, valid_idx) in enumerate(splits.split(total_set)):
    if(fold==0):
        continue
    # if (fold == 1):
    #     continue

    # if(fold==2):
    #     continue
    patience = 30  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, path=save_path + '/checkpoint.pth',
                                   verbose=True)  # 关于 EarlyStopping 的代码可先看博客后面的内容
    start_time = time.time()
    best_acc = 0.0

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


    criterion, model, optimizer = create_optimizer(load_model())

    # Training
    for epoch in range(h_epochs):
        print('--------------------------------------------------------')

        y_true, y_pred = [], []
        model.eval()
        vrunning_loss = 0.0
        vrunning_corrects = 0
        num_samples = 0

        for inputs, inputs_mask,labels in valid_loader:
            inputs = inputs.to(device)
            inputs_mask = inputs_mask.to(device)

            labels = labels.to(device)
            # optimizer.zero_grad()

            with torch.no_grad():
                outputs = model(inputs,inputs_mask)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # vrunning_loss += loss.item() * inputs.size(0)
            vrunning_corrects += (preds == labels).sum()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            num_samples += preds.size(0)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        vepoch_loss = vrunning_loss / num_samples
        vepoch_acc = (vrunning_corrects.double() * 100) / num_samples
        print(classification_report(y_true, y_pred, target_names=['非标准', '标准'], digits=3))

        # acc += torch.eq(predict_y, val_labels.to(device)).sum().item()


        if vepoch_acc > best_acc:
            best_acc = vepoch_acc
            # torch.save(model.state_dict(), save_path + "/resNet50_%d_%02d_%.3f.pth" % (fold,epoch,vepoch_acc))
        print('\t\t Validation({}) - Loss: {:.4f}, Acc: {:.4f}  Best Acc: {:.4f}'.format(epoch, vepoch_loss, vepoch_acc,best_acc))
