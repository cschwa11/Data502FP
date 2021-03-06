# -*- coding: utf-8 -*-
"""Downloading Data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1x7N0TQ7tszvjs-T4Y5fcIA3u5TYCTVPV
"""

!git clone https://github.com/cschwa11/Data502FP.git
import sys
sys.path.append('/content/Data502FP/models')
sys.path.append('/content/Data502FP/')
sys.path.append('/content/Data502FP/DFF/Pretrained')
sys.path.append('/content/Data502FP/DexiNed_CS')

!pip install kornia

import torch
import torchvision
import torchvision.transforms as transforms

import os
os.mkdir('/content/CIFAR10')

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=2)

CF10_img_lab=[]
with torch.no_grad():
  for batch_idx, (inputs, labels) in enumerate(testloader):
    class_label=str(labels).replace('tensor([','')
    class_label=class_label.replace('])','')
    image_path='/content/CIFAR10/image'+class_label+'_'+str(batch_idx)
    torchvision.utils.save_image(inputs,image_path+'.jpg')
    CF10_img_lab.append([(image_path.replace('/content/CIFAR10/image',''),labels)])
    torch.save(labels,image_path+'.pt')

#CIFAR-100
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=2)

CF100_img_lab=[]
with torch.no_grad():
  for batch_idx, (inputs, labels) in enumerate(testloader):
    class_label=str(labels).replace('tensor([','')
    class_label=class_label.replace('])','')
    image_path2='/content/data/cifar-100-python/test'+class_label+'_'+str(batch_idx)
    torchvision.utils.save_image(inputs,image_path2+'.jpg')
    CF100_img_lab.append([(image_path.replace('/content/data/cifar-100-python/test',''),labels)])
    torch.save(labels,image_path2+'.pt')

#SVHN
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.SVHN(
    root='./data', split='test', download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=2)

SVHN_img_lab=[]
with torch.no_grad():
  for batch_idx, (inputs, labels) in enumerate(testloader):
    class_label=str(labels).replace('tensor([','')
    class_label=class_label.replace('])','')
    image_path3='/content/data/test_32x32.mat'+class_label+'_'+str(batch_idx)
    torchvision.utils.save_image(inputs,image_path3+'.jpg')
    CF100_img_lab.append([(image_path.replace('/content/data/test_32x32.mat',''),labels)])
    torch.save(labels,image_path3+'.pt')