import os
import torch

import time
import numpy as np
from sklearn.metrics import classification_report

from PIL import Image
from sklearn.model_selection import KFold
from torch import nn
from torch import optim
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms
from model.resnet_v2 import resnet50_v2
from model.resnet_v1_dual_tsne import MyEnsemble
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

from pytorchtools import EarlyStopping
import random
import MyDataset
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

val_test_batch_size = 1
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


save_path = '/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/12_cross_model/2_grey/dual_28_0.00020'
os.makedirs(save_path, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_model():
    modelA = resnet50_v2()
    modelB = resnet50_v2()

    model = MyEnsemble(modelA, modelB, nb_classes=2)
    model_weight_path = "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/12_cross_model/resnet/dual_erase_28_0.00020/resNet50_1_03_92.148.pth"

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



for fold, (train_idx, valid_idx) in enumerate(splits.split(total_set)):
    if(fold==0):
        continue
    if(fold==1):
        continue


    # Train and val samplers
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)


    valid_loader = torch.utils.data.DataLoader(
        total_set, batch_size=val_test_batch_size, sampler=valid_sampler)


    criterion, model, optimizer = create_optimizer(load_model())




    y_true, y_pred = [], []
    embs = []
    labels = []

    model.eval()

    for inputs, inputs_mask,label in valid_loader:
        inputs = inputs.cuda()
        inputs_mask = inputs_mask.cuda()
        label = label.cuda()

        emb,outputs = model(inputs,inputs_mask)
        embs.append(emb.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

    embs = np.concatenate(embs)
    labels = np.concatenate(labels)
    tsne = TSNE(n_components=2, learning_rate=200, metric='cosine')
    tsne.fit_transform(embs)
    outs_2d = np.array(tsne.embedding_)

    css4 = list(mcolors.CSS4_COLORS.keys())
    # 我选择了一些较清楚的颜色，更多的类时也能画清晰
    color_ind = [2, 7, 9, 10, 11, 13, 14, 16, 17, 19, 20, 21, 25, 28, 30, 31, 32, 37, 38, 40, 47, 51,
                 55, 60, 65, 82, 85, 88, 106, 110, 115, 118, 120, 125, 131, 135, 139, 142, 146, 147]
    css4 = [css4[v] for v in color_ind]
    for lbi in range(10):
        temp = outs_2d[labels == lbi]
        plt.plot(temp[:, 0], temp[:, 1], '.', color=css4[lbi])
    plt.title('feats dimensionality reduction visualization by tSNE,test data')
    plt.show()
