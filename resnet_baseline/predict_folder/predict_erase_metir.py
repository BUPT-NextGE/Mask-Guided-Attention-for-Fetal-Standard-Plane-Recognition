import os
import json
import torch
from torchvision import transforms, datasets
from model.resnet_v2 import resnet50_v2
from sklearn.metrics import classification_report
import random
from PIL import Image

import numpy as np
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(40)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
training_transforms = transforms.Compose([transforms.Resize((224,224), Image.LANCZOS),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

batch_size = 64
image_path = os.path.join("/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/9_cross/fold_0/origin/val")  # flower data set path
validate_dataset = datasets.ImageFolder(root=image_path,
                                        transform=training_transforms)
flower_list = validate_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
json_str = json.dumps(cla_dict, indent=4)
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False)
net = resnet50_v2(num_classes=2).to(device)
weights_path = "/mnt/DataDrive164/lirj/medical/medical_dataset/21_11/05/12_cross_model/2_grey/resnet_erase_v2/resnet50_0.00015_fold3/resNet50_0_15_94.226.pth"
net.load_state_dict(torch.load(weights_path, map_location=device))
y_true, y_pred = [], []
net.eval()
acc = 0.0
with torch.no_grad():
    for val_images,val_labels in validate_loader:
        outputs = net(val_images.to(device))
        predict_y = torch.max(outputs, dim=1)[1]
        acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        y_true.extend(val_labels.cpu().numpy())
        print(y_true)
        y_pred.extend(predict_y.cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
print(classification_report(y_true, y_pred, target_names=['非标准', '标准'], digits=3))
val_accurate = acc / val_num
print(val_accurate)