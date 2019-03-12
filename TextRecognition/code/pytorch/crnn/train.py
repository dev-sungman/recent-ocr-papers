import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from  torch.autograd import Variable

import os
import random
import sys
import argparse

from data_loader import CrnnDataLoader
from models import CRNN

from tqdm import tqdm
from data_transform import Resize, Rotation, ToTensor

from collate_fn import text_collate

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='CRNN pytorch')
    
    # data configurations
    parser.add_argument('--dataroot', type=str, default="", help='Path to dataset')

    # input image size
    parser.add_argument('--input_size', type=str, default="320x32", help='Input size')

    # batch size
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')

    # learning rate
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    # epochs
    parser.add_argument('--epochs', type=int, default=100, help='Number of epoch')

    return parser.parse_args(argv)

def main(args):
    
    # cuda check
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # argument handling
    input_size = [int(x) for x in args.input_size.split('x')]

    # random seed
    random.seed(random.randint(1, 10000))

    # for faster training
    cudnn.banchmark = True
    cudnn.fastest = True
    
    # train transformation
    transform = transforms.Compose([
        Resize(size=(input_size[0], input_size[1])),
        ToTensor()
        ])

    # train dataset
    data = CrnnDataLoader(data_path=args.dataroot, mode="train", transform=transform)

    # model load
    nclass = data.cls_len()
    net = CRNN(nclass)
    
    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr,weight_decay=5e-4)

    # loss_function -> CTCLoss
    criterion = nn.CTCLoss()

    # epoch 
    best_acc = 0
    epoch = 0

    while epoch < args.epochs:
        data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, num_workers=4, shuffle=True)
        iterator = tqdm(data_loader)
        iter_count = 0
       
        ''' TODO: CTC LOSS '''
        for sample in iterator:
            optimizer.zero_grad()
            imgs = Variable(sample["img"])
            labels = Variable(sample["seq"]).view(-1)
            label_lens = Variable(sample["seq_len"]).view(-1)
            
            if device == 'cuda':
                imgs = imgs.cuda()

            preds = net(imgs).cpu()

            pred_lens = Variable(torch.Tensor(preds.size(0)).int())
            
            print("preds:", preds.shape)
            print("labels:", labels.shape)
            print("pred_lens", pred_lens.shape)
            print("label_lens", label_lens.shape)
            
            loss = criterion(preds, labels, pred_lens, label_lens)
            loss.backward()
            optimizer.step()
            status = "epoch: {}; loss: {}".format(epoch, loss.data[0])

        epoch += 1




if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

