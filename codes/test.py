import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import json
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from torch import  optim

CLUSTERS = json.load(open('filter_means_v3.json'))
# PATH = './cnn_v3.pth'
PATH = './cifar_net_gpu.pth'
FILTER_INDEX = json.load(open('filter_priority.json'))

class VGG19(nn.Module):
    '''
      Creates VGG19 acrh model
    '''

    def __init__(self, num_classes=10, init_weights=True, num=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            #1
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #2
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #3
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            #4.png
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #5
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #6
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #7
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #8
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #9
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #10
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #11
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #12
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #13
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #
            nn.AvgPool2d(kernel_size=1,stride=1),
            # nn.AdaptiveAvgPool2d((7,7)),
            )
        self.classifier = nn.Sequential(
            #14
            nn.Linear(512,4096),
            nn.ReLU(True),
            nn.Dropout(),
            #15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            #16
            nn.Linear(4096,num_classes),
            )
        #self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
#        print(out.shape)
        out = out.view(out.size(0), -1)
#        print(out.shape)
        out = self.classifier(out)
#        print(out.shape)
        return out


def prune(model_dict, cluster):
    filters = []
    for clu in cluster:
        filters.extend(CLUSTERS[str(clu)])
    newwei1 = np.array(model_dict['features.32.weight'].cpu())
    newwei2 = np.array(model_dict['features.34.weight'].cpu())
    newbias = np.array(model_dict['features.32.bias'].cpu())
    filters = sorted(filters, reverse=True)
    nw1 = []
    nw2 = []
    nb = []
    for filter in filters:
        # newwei1 = np.delete(newwei1, filter, 0)
        # newbias = np.delete(newbias, filter, 0)
        # newwei2 = np.delete(newwei2, filter, 1)
        nw1.append(newwei1[filter])
        nw2.append(newwei2[:, filter])
        nb.append(newbias[filter])
    nw2 = np.array(nw2)
    model_dict['features.32.weight'] = torch.Tensor(np.array(nw1))
    model_dict['features.34.weight'] = torch.Tensor(nw2.transpose((1,0,2,3)))
    model_dict['features.32.bias'] = torch.Tensor(np.array(nb))
    return model_dict


def prune_filter(label, threshold, model_dict):
    filters = []
    for filter, conf in FILTER_INDEX[label]:
        if conf > threshold:
            filters.append(filter)
    newwei1 = np.array(model_dict['features.32.weight'].cpu())
    newwei2 = np.array(model_dict['features.34.weight'].cpu())
    newbias = np.array(model_dict['features.32.bias'].cpu())
    filters = sorted(filters, reverse=True)
    nw1 = []
    nw2 = []
    nb = []
    for filter in filters:
        # newwei1 = np.delete(newwei1, filter, 0)
        # newbias = np.delete(newbias, filter, 0)
        # newwei2 = np.delete(newwei2, filter, 1)
        nw1.append(newwei1[filter])
        nw2.append(newwei2[:, filter])
        nb.append(newbias[filter])
    nw2 = np.array(nw2)
    # print(nw2.shape)
    model_dict['features.32.weight'] = torch.Tensor(np.array(nw1))
    model_dict['features.34.weight'] = torch.Tensor(nw2.transpose((1,0,2,3)))
    model_dict['features.32.bias'] = torch.Tensor(np.array(nb))
    return model_dict


clu_label = {6:[2],7:[5,3],2:[4,7],4:[9],1:[8],3:[4],8:[1,8],5:[6],0:[0],9:[8]}
pretrained_dict = torch.load(PATH)
# cut_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}#filter out unnecessary keys
# cut_dict = prune(cut_dict, 6)
cluster = [1,8,9]
num = 0
for clu in cluster:
    num += len(CLUSTERS[str(clu)])

model = VGG19(num = num)
# model = VGG19()
# model_dict = model.state_dict()
pretrained_dict = prune(pretrained_dict, cluster)
model.load_state_dict(pretrained_dict)

use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
criterion = nn.CrossEntropyLoss()

model.eval()
eval_loss = 0
eval_acc = 0
count = 0
for data in test_loader:
    img, label = data
    if label[0] != 8:
        continue
    count +=1
    if use_gpu:
        img = Variable(img, volatile=True).cuda()
        label = Variable(label, volatile=True).cuda()
    else:
        img = Variable(img, volatile=True)
        label = Variable(label, volatile=True)
    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / count, eval_acc / count))
