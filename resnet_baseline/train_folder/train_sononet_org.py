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
from model.sononet_org import SonoNet
from pytorchtools import EarlyStopping
import random
from sklearn.metrics import classification_report
from model.resnet_v1_dual  import MyEnsemble

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



total_set = datasets.ImageFolder(data_dir, transform=training_transforms)

# Defining folds
splits = KFold(n_splits = kfolds, shuffle = True, random_state = 42)
train_labels = {value: key for (key, value) in total_set.class_to_idx.items()}

print(len(train_labels))
print(train_labels)
save_path = '/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/12_cross_model/2_grey/resnet_erase_v2_seed02_1_9/resnet50_0.0002_fold3'
os.makedirs(save_path, exist_ok=True)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")



def load_model():
    model = SonoNet()
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

    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    return criterion.to(device), model.to(device), optimizer

# Variables to store fold scores
train_acc = []
test_top1_acc = []
test_top5_acc = []
test_precision = []
test_recall = []
test_f1 = []
times = []

for fold, (train_idx, valid_idx) in enumerate(splits.split(total_set)):
    if(fold==0):
        continue
    if (fold == 1):
        continue
    patience = 30  # ???????????????????????????20??????????????????????????????????????????????????????????????????????????????????????????
    early_stopping = EarlyStopping(patience, path=save_path + '/checkpoint.pth',
                                   verbose=True)  # ?????? EarlyStopping ???????????????????????????????????????
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

        model.train()
        y_true, y_pred = [], []

        running_loss = 0.0
        running_corrects = 0
        trunning_corrects = 0



        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum()
            trunning_corrects += preds.size(0)

        epoch_loss = running_loss / trunning_corrects
        epoch_acc = (running_corrects.double() * 100) / trunning_corrects
        train_acc.append(epoch_acc.item())

        print('\t\t Training: Epoch({}) - Loss: {:.4f}, Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
        # Validation
        model.eval()
        vrunning_loss = 0.0
        vrunning_corrects = 0
        num_samples = 0

        for data, labels in valid_loader:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                outputs = model(data)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            vrunning_loss += loss.item() * data.size(0)
            vrunning_corrects += (preds == labels).sum()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            num_samples += preds.size(0)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        vepoch_loss = vrunning_loss / num_samples
        vepoch_acc = (vrunning_corrects.double() * 100) / num_samples
        print(classification_report(y_true, y_pred, target_names=['?????????', '??????'], digits=3))

        if vepoch_acc > best_acc:
            best_acc = vepoch_acc
            torch.save(model.state_dict(), save_path + "/resNet50_%d_%02d_%.3f.pth" % (fold,epoch,vepoch_acc))
        print('\t\t Validation({}) - Loss: {:.4f}, Acc: {:.4f}  Best Acc: {:.4f}'.format(epoch, vepoch_loss, vepoch_acc,best_acc))



        # early_stopping(vepoch_loss, model)
        # if early_stopping.early_stop:
        #     early_stopping.early_stop = False
        #     early_stopping.counter = 0
        #     print("Early stopping")
        #     break