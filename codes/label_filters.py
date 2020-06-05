import torch
import torch.nn as nn
from torch import  optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
import os
from deepDream import *
import json

train_dataset = datasets.CIFAR10('./data', train=True, transform=transforms.ToTensor(), download=False)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
sample_num = 100

samples = {}
for i, data in tqdm(enumerate(train_loader, 1)):
    img, label = data
    label = int(label)
    if label not in samples:
        samples[label] = []
    if len(samples[label]) >= sample_num:
        continue
    samples[label].append(img)
    flag = 0
    for j in range(sample_num):
        try:
            if len(samples[j]) >= sample_num:
                flag += 1
        except:
            break
    if flag == sample_num:
        break
# for k, v in samples.items():
#     print(len(samples[k]))

model = DeepDream()
filter_clusters = json.load(open('filter_means_v3.json'))
cluser_references = {}
cou=0
for k, filters in filter_clusters.items():
    cluser_references[k] = {}
    for filter in filters:
        print('filter', cou)
        cou+=1
        filter_preference = {}
        for label, sps in samples.items():
            y = 0
            for sample in sps:
                y += abs(torch.sum(model.label_img(sample, label)[0][filter]))
            y /=100
            filter_preference[label] = float(y)
        filter_preference['prefer'] = max(filter_preference, key=lambda i:filter_preference[i])
        cluser_references[k][filter]=filter_preference
d = open('yils_v4.json', 'w')
try:
    json.dump(cluser_references, d)
## { 9: [ {1:,0.9, ...}, {  }, {  } ], [], ...      }
except:
    for k, v in cluser_references.items():
        d.write(k)
        d.write('\t')
        d.write(str(v))
        d.write('\n')
d.close()

filter_num = 512
filter_label = {}
for filter in range(0,filter_num ):
    for label, sps in samples.items():
        y = 0
        for sample in sps:
            y += abs(torch.sum(model.label_img(sample, label)[0][filter]))
        y /=100
        filter_preference[label] = float(y)
    filter_preference['prefer'] = max(filter_preference, key=lambda i:filter_preference[i])
    filter_label[filter] = filter_preference

d = open('filter_label.json', 'w')
try:
    json.dump(filter_label, d)
## { 9: [ {1:,0.9, ...}, {  }, {  } ], [], ...      }
except:
    for k, v in filter_label.items():
        d.write(k)
        d.write('\t')
        d.write(str(v))
        d.write('\n')
d.close()
