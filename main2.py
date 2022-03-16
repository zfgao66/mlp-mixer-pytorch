'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import logging
import torchvision
import torchvision.transforms as transforms
from mlp_mixer_pytorch import MLPMixer
import os
import argparse

from utils import get_dataset,train_model,test_model

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--data_path', type=str, default='/mnt/zfgao')
    parser.add_argument('--model', type=str, default='Mixer', help='model')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_model_pth', type=str, default='/mnt/zfgao/checkpoint/MLP_Mixer')
    parser.add_argument('--load_model_pth', type=str, default='')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr',type=float, default=0.001)
    parser.add_argument('--log_file',type=str, default='cifar_logs')
    
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--depth', type=int, default=12)
    

    args = parser.parse_args()

    logging.basicConfig(
            filename=args.log_file,
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO)

    logger = logging.getLogger(__name__)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ##################### dataset
    logger.info("'==> Preparing data..")
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = get_dataset(args.dataset, args.data_path)
    trainloader = torch.utils.data.DataLoader(dst_train,batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=args.batch_size, shuffle=False)
    ###################### Model
    logger.info("==> Building model..")
    net = MLPMixer(image_size=32,channels=3, patch_size=2,dim=512,depth=12,num_classes=10)
    net = net.to(args.device)
    if args.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    if args.load_model_pth:
        logger.info('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(args.load_model_pth, 'model.pth'))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    train_model(args,net,trainloader,testloader,optimizer, criterion)
    logger.info("Finished Training ...")
    # test_model(args,net, testloader, criterion)
    # logger.info("Finished Testing ...")
    # scheduler.step()





if __name__ == '__main__':
    main()