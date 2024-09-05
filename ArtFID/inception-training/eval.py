import argparse
import os
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
import torchvision
from torchvision import transforms as T
import numpy as np
import os
import glob
import wandb
from tqdm import tqdm
from PIL import Image

import datasets
import networks
import inception
import lr_scheduler
import utils


def main():
    # ----------
    # args
    # ----------
    parser = argparse.ArgumentParser()
    # Names and Paths
    parser.add_argument('--name', type=str, default='model')
    parser.add_argument('--group', type=str, default='art_classifier')
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--dataset_train_path', type=str)
    parser.add_argument('--dataset_val_path', type=str)
    parser.add_argument('--artist_class_weights_path')
    parser.add_argument('--style_class_weights_path')
    parser.add_argument('--class_weight_path', type=str)
    parser.add_argument('--pretrained_weights_path', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.set_defaults(cuda=True)
    args = parser.parse_args()


    artist_class_weights = torch.from_numpy(np.load(args.artist_class_weights_path).astype(np.float32))
    style_class_weights = torch.from_numpy(np.load(args.artist_class_weights_path).astype(np.float32))

    device = torch.device('cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu')
    model = inception.Inception3(num_classes1=artist_class_weights.shape[0],
                                 num_classes2=style_class_weights.shape[0]).to(device)

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    artist_val_acc, style_val_acc, artist_val_acc_top5, style_val_acc_top5 = validate(model, device, args)
    print('artist_val_acc:', artist_val_acc)
    print('style_val_acc:', style_val_acc)
    print('artist_val_acc_top5:', artist_val_acc_top5)
    print('style_val_acc_top5:', style_val_acc_top5)


def validate(model, device, args):
    val_transforms = T.Compose([
        #T.CenterCrop(size=(256, 256)),
        T.ToTensor(),
        #T.Normalize(mean=0.5, std=0.5),
    ])
    val_dataset = datasets.H5Dataset(path_to_h5_file=args.dataset_val_path, transforms=val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    model.eval()

    artist_val_acc = 0.0
    artist_val_count = 0
    style_val_acc = 0.0
    style_val_count = 0

    artist_val_acc_top5 = 0.0
    style_val_acc_top5 = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x = batch['image'].to(device)
            artist_label = batch['artist_label'].to(device)
            style_label = batch['style_label'].to(device)

            artist_pred, style_pred = model(x)
            
            artist_val_acc_, artist_val_acc_top5_ = utils.compute_accuracy(artist_pred, artist_label)
            if artist_val_acc_ != -1:
                artist_val_acc += artist_val_acc_
                artist_val_acc_top5 += artist_val_acc_top5_
                artist_val_count += 1 

            style_val_acc_, style_val_acc_top5_ = utils.compute_accuracy(style_pred, style_label)
            if style_val_acc_ != -1:
                style_val_acc += style_val_acc_
                style_val_acc_top5 += style_val_acc_top5_
                style_val_count += 1 


    artist_val_acc /= artist_val_count
    style_val_acc /= style_val_count

    artist_val_acc_top5 /= artist_val_count
    style_val_acc_top5 /= style_val_count

    model.train()
    return artist_val_acc, style_val_acc, artist_val_acc_top5, style_val_acc_top5


if __name__ == '__main__':
    main()
