'''Train TinyImageNetDataset with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import imageio
import numpy as np
import os

from collections import defaultdict
from torch.utils.data import Dataset

from tqdm.autonotebook import tqdm

# Define main data directory
DATA_DIR = 'tiny-imagenet-200' # Original images come in shapes of [3,64,64]

# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
VALID_DIR = os.path.join(DATA_DIR, 'val')

# Create separate validation subfolders for the validation images based on
# their labels indicated in the val_annotations txt file
val_img_dir = os.path.join(VALID_DIR, 'images')

# Open and read val annotations text file
fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
data = fp.readlines()

# Create dictionary to store img filename (word 0) and corresponding
# label (word 1) for every line in the txt file (as key value pair)
val_img_dict = {}
for line in data:
    words = line.split('\t')
    val_img_dict[words[0]] = words[1]
fp.close()

# Create subfolders (if not present) for validation images based on label,
# and move images into the respective folders
for img, folder in val_img_dict.items():
    newpath = (os.path.join(val_img_dir, folder))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    if os.path.exists(os.path.join(val_img_dir, img)):
        os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))


# Define Data Loader for Pytorch usage of Tiny Imagenet
def generate_dataloader(data, name, transform):
    if data is None: 
        return None
    
    # Read image files to pytorch dataset using ImageFolder, a generic data 
    # loader where images are in format root/label/filename
    # See https://pytorch.org/vision/stable/datasets.html
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=tranforms.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    # Set options for device
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}
    
    # Wrap image dataset (defined above) in dataloader 
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=(name=="train"), 
                        **kwargs)
    
    return dataloader
    
    
parser = argparse.ArgumentParser(description='PyTorch TinyImageNetDataset Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

					
# args=[] added to fix error
args = parser.parse_args(args=[])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_data = transforms.Compose([
    transforms.Resize(256), # Resize images to 256 x 256
                transforms.CenterCrop(224), # Center crop image
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # Converting cropped images to tensors
                transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                            std=[0.5, 0.5, 0.5])
])


trainloader = generate_dataloader(
    root=TRAIN_DIR, train=True, download=False, transform=transform_data)


testloader = generate_dataloader(
    root=TRAIN_DIR, train=False, download=False, transform=transform_data)


classes = ('n02124075','n04067472','n04540053','n04099969','n07749582','n01641577','n02802426','n09246464','n07920052','n03970156','n03891332','n02106662','n03201208','n02279972','n02132136','n04146614','n07873807','n02364673','n04507155','n03854065','n03838899','n03733131','n01443537','n07875152','n03544143','n09428293','n03085013','n02437312','n07614500','n03804744','n04265275','n02963159','n02486410','n01944390','n09256479','n02058221','n04275548','n02321529','n02769748','n02099712','n07695742','n02056570','n02281406','n01774750','n02509815','n03983396','n07753592','n04254777','n02233338','n04008634','n02823428','n02236044','n03393912','n07583066','n04074963','n01629819','n09332890','n02481823','n03902125','n03404251','n09193705','n03637318','n04456115','n02666196','n03796401','n02795169','n02123045','n01855672','n01882714','n02917067','n02988304','n04398044','n02843684','n02423022','n02669723','n04465501','n02165456','n03770439','n02099601','n04486054','n02950826','n03814639','n04259630','n03424325','n02948072','n03179701','n03400231','n02206856','n03160309','n01984695','n03977966','n03584254','n04023962','n02814860','n01910747','n04596742','n03992509','n04133789','n03937543','n02927161','n01945685','n02395406','n02125311','n03126707','n04532106','n02268443','n02977058','n07734744','n03599486','n04562935','n03014705','n04251144','n04356056','n02190166','n03670208','n02002724','n02074367','n04285008','n04560804','n04366367','n02403003','n07615774','n04501370','n03026506','n02906734','n01770393','n04597913','n03930313','n04118538','n04179913','n04311004','n02123394','n04070727','n02793495','n02730930','n02094433','n04371430','n04328186','n03649909','n04417672','n03388043','n01774384','n02837789','n07579787','n04399382','n02791270','n03089624','n02814533','n04149813','n07747607','n03355925','n01983481','n04487081','n03250847','n03255030','n02892201','n02883205','n03100240','n02415577','n02480495','n01698640','n01784675','n04376876','n03444034','n01917289','n01950731','n03042490','n07711569','n04532670','n03763968','n07768694','n02999410','n03617480','n06596364','n01768244','n02410509','n03976657','n01742172','n03980874','n02808440','n02226429','n02231487','n02085620','n01644900','n02129165','n02699494','n03837869','n02815834','n07720875','n02788148','n02909870','n03706229','n07871810','n03447447','n02113799','n12267677','n03662601','n02841315','n07715103','n02504458'
)

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
CUDA_LAUNCH_BLOCKING=1

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

# Changed to 80 epochs for speed reasons
for epoch in range(start_epoch, start_epoch+80):
    train(epoch)
    test(epoch)
    scheduler.step()
