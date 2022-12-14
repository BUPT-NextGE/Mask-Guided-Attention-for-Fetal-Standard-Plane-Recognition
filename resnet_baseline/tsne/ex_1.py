from model.resnet_v2_tsne import resnet50_v2
from model.resnet_v1_tsne import resnet50_v1

import torch
from torchvision import transforms, datasets
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50_v1(num_classes=2).to(device)

# load model weights
weights_path = "/mnt/DataDrive164/lirj/medical/weights_backup/resnet_org/resNet50_2_0.947.pth"
batch_size = 16
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
data_transform = {
    "train": transforms.Compose([transforms.Resize((224,224)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize((224,224)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
image_path = '/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_cross/fold_2/origin/val/'

model.load_state_dict(torch.load(weights_path, map_location=device))
validate_dataset = datasets.ImageFolder(root=image_path,
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw)

model.eval()
embs = []
labels = []
for image, target in validate_loader:
    image, target = image.cuda(), target.cuda()
    print(type(target))

    emb,output = model(image)
    embs.append(emb.data.cpu().numpy())
    labels.append(target.data.cpu().numpy())

embs = np.concatenate(embs)
labels = np.concatenate(labels)
tsne = TSNE(n_components=2, learning_rate=200, metric='cosine')
tsne.fit_transform(embs)
outs_2d = np.array(tsne.embedding_)


css4 = list(mcolors.CSS4_COLORS.keys())
#我选择了一些较清楚的颜色，更多的类时也能画清晰
color_ind = [2,7,9,10,11,13,14,16,17,19,20,21,25,28,30,31,32,37,38,40,47,51,
         55,60,65,82,85,88,106,110,115,118,120,125,131,135,139,142,146,147]
css4 = [css4[v] for v in color_ind]
for lbi in range(10):
    temp = outs_2d[labels==lbi]
    plt.plot(temp[:,0],temp[:,1],'.',color=css4[lbi])
plt.title('feats dimensionality reduction visualization by tSNE,test data')
plt.show()
