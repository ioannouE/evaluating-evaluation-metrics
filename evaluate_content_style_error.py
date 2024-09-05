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
from skimage.util import random_noise
from skimage.transform import swirl, warp

import lpips

from vgg import Vgg16
import utils
import random

np.random.seed(42)


# https://github.com/safwankdb/ReCoNet-PyTorch/blob/master/testwarp.py

def get_subdirectories(directory_path):
    subdirectories = []
    for entry in os.scandir(directory_path):
        if entry.is_dir():
            subdirectories.append(entry.name)
    return subdirectories


def visualize_image(tensor_image):
    # Convert tensor to PIL Image for visualization
    image = transforms.ToPILImage()(tensor_image)

    # Display the image
    plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
    plt.imshow(image)
    plt.axis('off')  # Remove axis labels
    # plt.show()
    plt.savefig('Rectangles_blur_.png',bbox_inches='tight',pad_inches = 0, dpi = 300)

def add_black_rectangles_tensor(image_tensor, num_rectangles=5, rectangle_size=(10, 10)):
    """
    Add black rectangles to an image tensor.

    :param image_tensor: Input image tensor.
    :param num_rectangles: Number of black rectangles to add.
    :param rectangle_size: Size of the rectangles (width, height).
    :return: Image tensor with black rectangles.
    """
    # Clone the image tensor to avoid modifying the original
    img_with_rectangles = image_tensor.clone()

    _, height, width = img_with_rectangles.shape

    for _ in range(num_rectangles):
        # Random top-left corner of the rectangle
        x = random.randint(0, width - rectangle_size[0])
        y = random.randint(0, height - rectangle_size[1])

        # Adding the black rectangle
        img_with_rectangles[:, y:y+rectangle_size[1], x:x+rectangle_size[0]] = 0

    return img_with_rectangles

def main():
    
    parser = argparse.ArgumentParser(description='parser for evaluating a model')
    parser.add_argument("--stylized", type=str, required=True,
                        help="folder that contains the images")
    parser.add_argument("--cuda", type=int, default=1, required=False,
                                  help="use cuda")
    parser.add_argument("--image-size", type=int, default=512,
                                  help="the image size")
    parser.add_argument("--style-image", type=str, required=False,
                                  help="the style image")
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

    mse_loss = torch.nn.MSELoss()
    lpips_sim = lpips.LPIPS(net='squeeze').to(device)
    vgg = Vgg16(requires_grad=False).to(device)

    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    # style = utils.load_image(args.style_image)
    # style = style_transform(style).to(device)
    # style = style.repeat(1, 1, 1, 1).to(device)
    
    # features_style = vgg(utils.normalize_batch(style))
    # gram_style = [utils.gram_matrix(y) for y in features_style]

    
    sum_content = 0.
    sum_style = 0.
    #############################################################################

    benchmark_dir = 'testing_images/tested_image' # 'benchmark_unstylized'
    # retrieve the original frames
    original_frames_path = benchmark_dir + '/*.png' # '/*.jpg'
    original_frames = []
    for img in sorted(glob.glob(original_frames_path)):
        # image = Image.open(img).convert('RGB')            
        # image = transforms.Resize((args.image_size,args.image_size))(image)
        # image = transforms.ToTensor()(image)
        # image = image.unsqueeze(0).to(device)
        original_frames.append(img)
    print('original frames: ', len(original_frames))

    styles_path = 'styles'
    # retrieve the original frames
    styles_path = styles_path + '/*.jpg'
    all_styles = []
    for img in sorted(glob.glob(styles_path)):
        # print(img)
        # image = Image.open(img).convert('RGB')            
        # image = transforms.Resize((args.image_size,args.image_size))(image)
        # image = transforms.ToTensor()(image)
        # image = image.unsqueeze(0).to(device)
        all_styles.append(img)
    print('all styles: ', len(all_styles))

    content_error = 0.
    style_error = 0.
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

            # salt and pepper
            # stylized_image = torch.tensor(random_noise(stylized_image, mode='s&p', amount=0.1*args.perturbation, clip=True))

            # gaussian noise
            stylized_image = torch.tensor(random_noise(stylized_image, mode='gaussian', mean=0, var=0.1*args.perturbation, clip=True))
            stylized_image = stylized_image.float()

            # implanted black rectangles
            # stylized_image = add_black_rectangles_tensor(stylized_image.clone(), 10+int(10*args.perturbation), (10+int(30*args.perturbation), 10+int(30*args.perturbation)))

            # swirl
            # image_np = stylized_image.permute(1, 2, 0).numpy()
            # swirled = swirl(image_np, radius=100 + (140 * args.perturbation), rotation=0, strength=1+int(args.perturbation), clip=True, center=(image_np.shape[1] / 2, image_np.shape[0] / 2), order=1)
            # stylized_image = torch.tensor(swirled).permute(2,0,1)


            # visualize_image(stylized_image)


        stylized_image = stylized_image.unsqueeze(0).to(device)
        stylized_images.append(stylized_image)

        # get the original content image
        # print(stylized_img)
        for content_img in original_frames:
        # for style in all_styles:
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

                # style_image = utils.load_image(style)
                # style_image = style_transform(style_image).to(device)
                # style_image = style_image.repeat(1, 1, 1, 1).to(device)


                features_org = vgg(content_image)
                features_stylized = vgg(stylized_image)
                content_error += mse_loss(features_stylized.relu2_2, features_org.relu2_2)
                print('Image: ', desired_part, ' content error: {:.4f}'.format(mse_loss(features_stylized.relu2_2, features_org.relu2_2)))
                # features_style = vgg(utils.normalize_batch(style_image))
                # gram_style = [utils.gram_matrix(y) for y in features_style]

    
                # style_loss = 0.
                # for ft_y, gm_s in zip(features_stylized, gram_style):
                #     gm_y = utils.gram_matrix(ft_y)
                #     style_loss += mse_loss(gm_y, gm_s)
                # style_error += style_loss

                break

    print('--------------------------------------------------------------')              
    # print('Average content error: {:.4f}'.format(content_error.item() / len(stylized_images)))
    print('Average content error: {:.4f}'.format(content_error.item()/len(stylized_images)), '  (Perturbation: {:.2f})'.format(args.perturbation))
    # print('Average style error: ', style_error.item()/len(stylized_images))
    sum_content += content_error/len(stylized_images)
    sum_style += style_error/len(stylized_images)
    print('--------------------------------------------------------------')  



    # content_error = 0.
    # style_error = 0.
    # for itr, (org, stylized) in enumerate(zip(original_frames, stylized_images)):
    #     print(stylized.name)
    #     features_org = vgg(org)
    #     features_stylized = vgg(stylized)
    #     content_error += mse_loss(features_stylized.relu2_2, features_org.relu2_2)

    #     style_loss = 0.
    #     for ft_y, gm_s in zip(features_stylized, gram_style):
    #         gm_y = utils.gram_matrix(ft_y)
    #         style_loss += mse_loss(gm_y, gm_s)
    #     style_error += style_loss

    # print('Average content error: ', content_error.item()/len(original_frames))
    # print('Average style error: ', style_error.item()/len(original_frames))
    # sum_content += content_error/len(original_frames)
    # sum_style += style_error/len(original_frames)
        
    
    # print('Average content error over all directories: ', round(sum_content.item()/len(INPUT_DIRS),4))
    # print('Average style error over all directories: ', round(sum_style.item()/len(INPUT_DIRS),8))
        

    
if __name__ == "__main__":
    main()