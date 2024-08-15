''' 
inferencing based on code adapted from Kumar Ayush, Burak Uzkent, Chenlin Meng, Kumar Tanmay, 
Marshall Burke, David Lobell, and Stefano Ermon. Geography-aware self-supervised learning. 
In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 10161–10170, 2021.
'''

import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from torch.utils.data import random_split
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchgeo.datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from tensorboardX import SummaryWriter
import torchvision.models as models 


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--data', metavar='DIR', type=str, default='BigEarthNet',
                    help='path to dataset')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet50)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')

parser.add_argument('--pretrained', default='pretrainedWeights/moco.pth.tar', type=str,
                    help='path to moco pretrained checkpoint')

parser.add_argument('--save-dir', default='extractedFeatures/moco/', type=str,
                    help='save location')


def main():
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        if len(args.save_dir) > 0:
            os.makedirs(args.save_dir)

    main_worker(args)


def calculate_stats(loader):
    mean = 0.0
    std = 0.0
    total_images = 0

    for i, batch in enumerate(loader):
        images = batch["image"].numpy()  # Channels-last für numpy
        batch_samples = images.shape[0]  # Anzahl der Bilder in der Batch
        images = images.reshape(batch_samples, images.shape[1],-1)  # Flache jedes Bild
        mean += np.sum(images.mean(axis=-1), axis=0)
        std += np.sum(images.std(axis=-1), axis=0)
        total_images += batch_samples
        if i == 20:
            break

    mean /= total_images
    std /= total_images
    return mean, std


class NormalizeDict:
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean=[mean], std=[std])
        self.to_tensor = transforms.ToTensor()
    
    def __call__(self, sample):
        image = sample['image']
        image = self.to_tensor(image)
        image = self.normalize(image)
        sample['image'] = image
        return sample


def main_worker(args):
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    model.fc = nn.Linear(model.fc.weight.size(1), 128)
    
    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        #for name, param in state_dict.items():
            #    print(f"Layer: {name}")

        for k in list(state_dict.keys()):
            # retain only encoder_q
            if k.startswith('module.encoder_q'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        #for name, param in state_dict.items():
            #    print(f"Layer: {name}")

        args.start_epoch = 0
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(args.pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(args.pretrained))

    model = model.to("cpu")

    
    train_dataset = torchgeo.datasets.BigEarthNet(root=args.data, split='train', bands='s2')
    val_dataset = torchgeo.datasets.BigEarthNet(root=args.data, split='val', bands='s2')
    test_dataset = torchgeo.datasets.BigEarthNet(root=args.data, split='test', bands='s2')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    
    mean, std = calculate_stats(train_loader)

    for loader, filename in zip([train_loader, val_loader, test_loader], ["train.json", "val.json", "test.json"]):
        inference(loader, model, filename, mean, std, args)



def inference(loader, model, filename, mean, std, args):
    data_to_save = {
        "feature_vectors": [],
        "labels": []
    }

    model.eval()
    print("num itarations: ", len(loader))
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader)):
            images = batch["image"].to("cpu")
            target = batch["label"].to("cpu")
            
            # normalization
            for c in [1, 2, 3]:
                images[:, c, :, :] = (images[:, c, :, :] - mean[c]) / std[c]

            img = torch.zeros(len(images), 3, 120, 120)
            img[:, 0] = images[:, 3]
            img[:, 1] = images[:, 2]
            img[:, 2] = images[:, 1]
            
            # RGB image test
            if i == 0:
            	numpy_array = img[2].permute(1, 2, 0).numpy()
            	rgb_image = (numpy_array - np.min(numpy_array)) / (np.max(numpy_array) - np.min(numpy_array))
            	plt.imshow(rgb_image)
            	plt.savefig("../exampleResults/plots/RGBimageBigEarthNetS2.jpg")

            output = model(img)

            if len(data_to_save["labels"]) == 0:
                data_to_save["labels"] = target.numpy().tolist()
                data_to_save["feature_vectors"] = output.numpy().tolist()
            else:
                data_to_save["labels"] += target.numpy().tolist()
                data_to_save["feature_vectors"] += output.numpy().tolist()
    
    with open(args.save_dir + filename, 'w') as f:
        json.dump(data_to_save, f, indent=4)



if __name__ == '__main__':
    main()
