import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import random
import sys
import argparse
from data_loader import CrnnDataLoader


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='CRNN pytorch')
    
    # data configurations
    parser.add_argument('--dataroot', type=str, default=None, help='Path to dataset')

    # input image size
    parser.add_argument('--input_size', type=str, default="320x32", help='Input size')

    # batch size
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')

    return parser.parse_args(argv)

def main(args):
    # argument handling
    input_size = [int(x) for x in args.input_size.split('x')]

    # random seed
    random.seed(random.randint(1, 10000))

    # for faster training
    cudnn.banchmark = True

    # train transformation
    transform = transforms.Compose([
        transforms.Resize(size=[input_size[0], input_size[1]])
        ])

    # train dataset
    data = CrnnDataLoader(data_path=args.dataroot, mode="train", transform=transform)

    # model load

    # optimizer

    # loss_function -> CTCLoss

    # epoch 
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

