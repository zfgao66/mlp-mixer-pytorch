from ast import arg
import numpy as np
import torch
import logging
from tqdm import tqdm
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate
from vanilla_train import VanillaTrain
from networks import MLPMixer
logger = logging.getLogger(__name__)
def get_dataset(dataset, data_path):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)  # no augmentation
        dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    else:
        exit('unknown dataset: %s'%dataset)

    
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test

def get_trainer(args, trainloader, testloader):
    logger.info("Using {}".format(args.model))

    # ----- Setting model
    if args.model == 'MLPMixer':
        model = eval(args.model)(in_channels=3, dim=512, num_classes=10, patch_size=args.patch_size, image_size=32, depth=args.depth, token_dim=args.token_dim, channel_dim=args.channel_dim).to(args.device)
    else:
        raise NotImplementedError
    
    
    optimizer = optim.SGD(model.parameters(), args.learning_rate, weight_decay=5e-4,momentum=0.9)

    if args.train_method == 'VanillaTrain':
        distiller = VanillaTrain(model, trainloader, testloader, 
                            optimizer, device=args.device)  
    else:
        raise NotImplementedError

    return distiller
# Training
# def train_model(args,model,trainloader,testloader,optimizer, criterion):
#     # logger.info(f"\nEpoch: {args.epoch}")
#     model.train()
#     train_loss = 0
#     # length_of_dataset = len(trainloader.dataset)
#     correct = 0
#     total = 0
#     for ep in range(args.epoch):
#         for batch_idx, (inputs, targets) in enumerate(trainloader):
#             inputs, targets = inputs.to(args.device), targets.to(args.device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets.view_as(predicted)).sum().item()
#         epoch_acc = correct / total
#         logger.info(f"Epoch: {ep+1}, Train Loss:{train_loss/(batch_idx+1)}, Accuracy: {epoch_acc}")
#         test_model(args, model, testloader, criterion)


# def test_model(args,model, testloader, criterion):
#     best_acc = 0.0
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(args.device), targets.to(args.device)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets.view_as(predicted)).sum().item()
#         eval_acc = correct / total
#         logger.info(f"Test Loss:{test_loss/(batch_idx+1)}, Accuracy: {eval_acc}")
            
#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': model.state_dict(),
#             'acc': acc,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, '/mnt/zfgao/checkpoint/Mixer/ckpt.pth')
#         best_acc = acc
