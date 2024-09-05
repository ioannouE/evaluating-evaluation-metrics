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
import csv

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

def apply_noise(image_tensor, noise_type, perturbation_level):
    device = image_tensor.device
    
    if noise_type == 'random_perturbation':
        perturbation = torch.randn(image_tensor.size(), device=device) * perturbation_level
        perturbed_image = image_tensor + perturbation
        return torch.clamp(perturbed_image, 0, 1)
    elif noise_type == 'blur':
        kernel_sizes = {0.1:5, 0.25:11, 0.5: 17, 1.0:23, 2.0: 29, 3.0:35}
        kernel_size = kernel_sizes.get(perturbation_level, 35)  # Default to 35 if not in dict
        gaussian_blur = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=perturbation_level*3)
        return torch.clamp(gaussian_blur(image_tensor), 0, 1)
    elif noise_type == 'salt_and_pepper':
        noisy = torch.tensor(random_noise(image_tensor.cpu().numpy(), mode='s&p', amount=0.1*perturbation_level, clip=True), device=device)
        return torch.clamp(noisy, 0, 1)
    elif noise_type == 'gaussian':
        noisy = torch.tensor(random_noise(image_tensor.cpu().numpy(), mode='gaussian', mean=0, var=0.1*perturbation_level, clip=True), device=device)
        return noisy.float()
    elif noise_type == 'rectangles':
        return utils.add_black_rectangles_tensor(image_tensor.clone(), 10+int(10*perturbation_level), (10+int(30*perturbation_level), 10+int(30*perturbation_level)))
    elif noise_type == 'swirl':
        image_np = image_tensor.cpu().permute(1, 2, 0).numpy()
        swirled = swirl(image_np, radius=100 + (140 * perturbation_level), rotation=0, strength=1+int(perturbation_level), clip=True, center=(image_np.shape[1] / 2, image_np.shape[0] / 2), order=1)
        return torch.tensor(swirled, device=device).permute(2,0,1)
    else:  # 'none' or default case
        return image_tensor

def is_random_noise(noise_type):
    return noise_type in ['random_perturbation', 'gaussian', 'salt_and_pepper', 'rectangles', 'swirl' ]

def visualize_and_save_image(tensor_image, save_path):
    # Convert tensor to PIL Image for visualization
    image = transforms.ToPILImage()(tensor_image)
    
    # Save the image
    image.save(save_path)

def main():
    parser = argparse.ArgumentParser(description='Evaluate content error for perturbed images')
    parser.add_argument("--image-dir", type=str, required=True, help="directory containing images")
    parser.add_argument("--cuda", type=int, default=1, help="use cuda")
    parser.add_argument("--image-size", type=int, default=512, help="the image size")
    parser.add_argument("--runs", type=int, default=5, help="number of runs for random noise types")
    parser.add_argument("--output-dir", type=str, default="perturbed_images", help="directory to save perturbed images")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print("Running on", device)

    mse_loss = torch.nn.MSELoss()
    vgg = Vgg16(requires_grad=False).to(device)

    perturbation_levels = [0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
    noise_types = ['random_perturbation','salt_and_pepper','blur','gaussian','rectangles','swirl']
    image_paths = glob.glob(os.path.join(args.image_dir, '*.png'))

    # results = {img_path: {noise_type: {level: 0 for level in perturbation_levels} for noise_type in noise_types} for img_path in image_paths}
    results = {img_path: {noise_type: {level: 0 for level in perturbation_levels} for noise_type in noise_types} for img_path in image_paths}

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    
    for img_path in image_paths:
        original_image = Image.open(img_path).convert('RGB')
        original_image = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])(original_image).unsqueeze(0).to(device)

        style_image = utils.load_image(img_path)
        style_image = style_transform(style_image).to(device)
        style_image = style_image.repeat(1, 1, 1, 1).to(device)

        # Calculate Gram matrices for the original image
        features_style = vgg(utils.normalize_batch(style_image))
        gram_style = [utils.gram_matrix(y) for y in features_style]


        for noise_type in noise_types:
            for perturbation in perturbation_levels:
                if perturbation == 0:
                    # Skip noise application for perturbation level 0
                    perturbed_image = original_image
                else:
                    if is_random_noise(noise_type):
                        total_style_error = 0
                        for run in range(args.runs):
                            perturbed_image = apply_noise(original_image.squeeze(0), noise_type, perturbation).unsqueeze(0)
                            features_perturbed = vgg(perturbed_image)
                            style_loss = 0.
                            for ft_y, gm_s in zip(features_perturbed, gram_style):
                                gm_y = utils.gram_matrix(ft_y)
                                style_loss += mse_loss(gm_y, gm_s)
                            total_style_error += style_loss.item()
                        
                            # Save the last run of the highest perturbation level
                            if perturbation == 3.0 and run == args.runs - 1:
                                save_path = os.path.join(args.output_dir, f"{os.path.basename(img_path)[:-4]}_{noise_type}_perturbed.png")
                                visualize_and_save_image(perturbed_image.squeeze(0).cpu(), save_path)
                    
                        avg_style_error = (total_style_error / args.runs) * 100
                    else:
                        perturbed_image = apply_noise(original_image.squeeze(0), noise_type, perturbation).unsqueeze(0)
                        features_perturbed = vgg(perturbed_image)
                        style_loss = 0.
                        for ft_y, gm_s in zip(features_perturbed, gram_style):
                            gm_y = utils.gram_matrix(ft_y)
                            style_loss += mse_loss(gm_y, gm_s)
                        avg_style_error = style_loss.item()
                        avg_style_error = avg_style_error * 100

                        # Save the highest perturbation level
                        if perturbation == 3.0:
                            save_path = os.path.join(args.output_dir, f"{os.path.basename(img_path)[:-4]}_{noise_type}_perturbed.png")
                            visualize_and_save_image(perturbed_image.squeeze(0).cpu(), save_path)

                    results[img_path][noise_type][perturbation] = avg_style_error
                    print(f'Image: {os.path.basename(img_path)}, Noise: {noise_type}, Perturbation: {perturbation:.2f}, Avg Style Error: {avg_style_error:.4f}')

    # Export results to CSV
    with open('style_error_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        for img_path in image_paths:
            # Write image name
            writer.writerow([os.path.basename(img_path)])
            
            # Write header
            header = ['Noise Level'] + noise_types
            writer.writerow(header)
            
            # Write data rows for this image
            for level in perturbation_levels:
                row = [level]
                for noise_type in noise_types:
                    style_error = results[img_path][noise_type][level]
                    row.append(f'{style_error:.4f}')
                writer.writerow(row)
            
            # Add an empty line between tables
            writer.writerow([])
    
    print("Results exported to style_error_results.csv")
    if args.output_dir:
        print(f"Perturbed images saved in {args.output_dir}")

if __name__ == "__main__":
    main()