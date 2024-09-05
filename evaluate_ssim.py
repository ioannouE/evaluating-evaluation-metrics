import argparse
import os
import sys

from skimage import io, transform
from torch.utils.data import DataLoader
import torch
torch.cuda.empty_cache()

import gc
gc.collect()

import re
import numpy as np
from scipy import misc
from PIL import Image
from torch.autograd import Variable
import glob
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

import lpips
#import pytorch_ssim
from piq import ssim, SSIMLoss
from skimage.util import random_noise

from skimage.transform import swirl, warp
import random
import utils


# https://github.com/safwankdb/ReCoNet-PyTorch/blob/master/testwarp.py


def main():
    
    parser = argparse.ArgumentParser(description='parser for evaluating a model')
    parser.add_argument("--stylized", type=str, required=True,
                        help="folder that contains the images")
    parser.add_argument("--cuda", type=int, default=1, required=False,
                                  help="use cuda")
    parser.add_argument("--image-size", type=int, default=512,
                                  help="the image size")
    parser.add_argument("--perturbation", type=float, default=0.1)
    parser.add_argument("--noise", type=int, default=0, help="add noise/perturbation to the image")
    
    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

     # set up midas
    device = torch.device("cuda" if args.cuda else "cpu")
    print("Device: ", torch.cuda.get_device_name(0))
    print("Running on ", device)

    # mse_loss = torch.nn.MSELoss()
    # lpips_sim = lpips.LPIPS(net='squeeze').to(device)
    # ssim_loss = pytorch_ssim.SSIM(window_size = 11)


    #############################################################################
    benchmark_dir = 'testing_images/fid_all' # 'benchmark_unstylized'
    # retrieve the original frames
    original_frames_path = benchmark_dir + '/*.jpg'
    original_frames = []
    for img in sorted(glob.glob(original_frames_path)):
        # image = Image.open(img).convert('RGB')            
        # image = transforms.Resize((args.image_size,args.image_size))(image)
        # image = transforms.ToTensor()(image)
        # image = image.unsqueeze(0).to(device)
        original_frames.append(img)
    print('original frames: ', len(original_frames))

    # styles_path = 'styles'
    # # retrieve the original frames
    # styles_path = styles_path + '/*.jpg'
    # all_styles = []
    # for img in sorted(glob.glob(styles_path)):
    #     print(img)
    #     # image = Image.open(img).convert('RGB')            
    #     # image = transforms.Resize((args.image_size,args.image_size))(image)
    #     # image = transforms.ToTensor()(image)
    #     # image = image.unsqueeze(0).to(device)
    #     all_styles.append(img)
    # print('all styles: ', len(all_styles))


    dist = 0.
    test_frames_path = args.stylized
    print(test_frames_path)
    stylized_images = []
    for stylized_img in sorted(glob.glob(test_frames_path + '/*.*')): 
        stylized_image = Image.open(stylized_img).convert('RGB')        
        stylized_image = transforms.Resize((args.image_size,args.image_size))(stylized_image)
        stylized_image = transforms.ToTensor()(stylized_image)
        if (args.noise):
            # perturbation_strength = args.perturbation
            # perturbation = torch.randn(stylized_image.size()) * perturbation_strength
            # perturbed_image = stylized_image + perturbation
            # perturbed_image = torch.clamp(perturbed_image, 0, 1)
            # stylized_image = perturbed_image

            # gaussian blur
            # kernel_sizes = {0.1:5, 0.25:11, 0.5: 17, 1.0:23, 2.0: 29, 3.0:35}
            # gaussian_blur = transforms.GaussianBlur(kernel_size=(kernel_sizes[args.perturbation],kernel_sizes[args.perturbation]), sigma=args.perturbation*3)
            # stylized_image = gaussian_blur(stylized_image)
            # stylized_image = torch.clamp(stylized_image, 0, 1)

            # salt and pepper
            # stylized_image = torch.tensor(random_noise(stylized_image, mode='s&p', amount=0.1*args.perturbation, clip=True))
            # stylized_image = torch.clamp(stylized_image, 0, 1)

            # gaussian noise
            # stylized_image = torch.tensor(random_noise(stylized_image, mode='gaussian', mean=0, var=0.1*args.perturbation, clip=True))
            # stylized_image = stylized_image.float()

            # implanted black rectangles
            # stylized_image = utils.add_black_rectangles_tensor(stylized_image.clone(), 10+int(10*args.perturbation), (10+int(30*args.perturbation), 10+int(30*args.perturbation)))

            # swirl
            image_np = stylized_image.permute(1, 2, 0).numpy()
            swirled = swirl(image_np, radius=100 + (140 * args.perturbation), rotation=0, strength=1+int(args.perturbation), clip=True, center=(image_np.shape[1] / 2, image_np.shape[0] / 2), order=1)
            stylized_image = torch.tensor(swirled).permute(2,0,1)
            
            # utils.visualize_image(stylized_image)

        stylized_image = stylized_image.unsqueeze(0).to(device)
        stylized_images.append(stylized_image)

        # get the original content image
        # print(stylized_img)
        for content_img in original_frames:
            base_filename = os.path.basename(content_img)
            name_without_extension, _ = os.path.splitext(base_filename)
            desired_part = name_without_extension[:-1]
            # print(desired_part)
            if desired_part in stylized_img:
                # print(stylized_img, "   -----    ", desired_part)
                content_image = Image.open(content_img).convert('RGB')            
                content_image = transforms.Resize((args.image_size,args.image_size))(content_image)
                content_image = transforms.ToTensor()(content_image)
                content_image = content_image.unsqueeze(0).to(device)
                
                dist += ssim(content_image, stylized_image)
                print('Image: ', desired_part, ' SSIM: {:.4f}'.format(ssim(content_image, stylized_image)))


    print('Average SSIM: {:.4f}'.format(dist.item()/len(stylized_images)))
    print('--------------------------------------------------------------')              





if __name__ == "__main__":
    main()