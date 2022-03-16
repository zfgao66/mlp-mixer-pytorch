import argparse
import logging
from random import shuffle
import os
from numpy import float32, float64
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
# from mlp_mixer_pytorch import MLPMixer
from utils import get_dataset,get_trainer


def main():
    ##################### config
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--data_path', type=str, default='/mnt/zfgao')
    parser.add_argument('--model', type=str, default='MLPMixer', help='model')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_model_pth', type=str, default='/mnt/zfgao/checkpoint/MLP_Mixer')
    parser.add_argument('--load_model_pth', type=str, default='')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate',type=float32, default=0.001)
    parser.add_argument('--log_file',type=str, default='cifar_logs')
    parser.add_argument('--train_method',type=str, default='VanillaTrain')
    
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--token_dim', type=int, default=128)
    parser.add_argument('--channel_dim', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=12)
    

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logger.info(f"Args:{args}")
    # logger.info(f"Total HyperPrameters:{args.__dict__{}}")
    logging.basicConfig(
        filename=args.log_file,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    logger = logging.getLogger(__name__)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ##################### dataset
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = get_dataset(args.dataset, args.data_path)
    trainloader = torch.utils.data.DataLoader(dst_train,batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=args.batch_size, shuffle=False)
    ##################### models
    logger.info("Begin Training Processing.")
    trainer = get_trainer(args, trainloader, testloader)
    trainer.get_parameters()
    # model = MLPMixer(image_size=32,channels=3, patch_size=2,dim=512,depth=12,num_classes=10)
    ##################### Training
    if args.model =='MLPMixer':
        if args.load_model_pth:
            logger.info("####### Check : Using Pre-trained Weight from {}".format(args.load_model_pth))
            trainer.model.load_state_dict(torch.load(args.load_model_pth))
        else:
            trainer.train_model(epochs=args.epoch, save_model=True, save_model_pth=os.path.join(args.save_model_pth,'trained_model.pth'))
    else:
        raise NotImplementedError
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    # model = model.to(args.device)
    # length_of_dataset = len(trainloader.dataset)
    # for epoch in range(args.epoch):
    #     epoch_loss = 0.0
    #     correct = 0
    #     for (data, label) in enumerate(trainloader):
    #         data = data.to(args.device)
    #         label = label.to(args.device)
    #         out = model(data)
    #         if isinstance(out, tuple):
    #                 out = out[0]
    #         pred = out.argmax(dim=1, keepdim=True)
    #         correct += pred.eq(label.view_as(pred)).sum().item()
    #         loss = criterion(out, label)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss
    #     epoch_acc = correct/length_of_dataset
    #     #Evaluate the model
    #     epoch_val_acc = 
            
    
    logger.info("Finished Training ...")

if __name__ == '__main__':
    main()