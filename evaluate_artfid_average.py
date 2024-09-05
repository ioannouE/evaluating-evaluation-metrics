import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

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
import shutil

# Add the ArtFID folder to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
artfid_dir = os.path.join(script_dir, 'ArtFID')
sys.path.append(artfid_dir)

from art_fid import art_fid
import tempfile
from torchvision.utils import save_image


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
    parser = argparse.ArgumentParser(description='Evaluate ArtFID for perturbed images')
    parser.add_argument("--image-dir", type=str, required=True, help="directory containing images")
    parser.add_argument("--cuda", type=int, default=1, help="use cuda")
    parser.add_argument("--image-size", type=int, default=512, help="the image size")
    parser.add_argument("--runs", type=int, default=5, help="number of runs for random noise types")
    parser.add_argument("--output-dir", type=str, help="directory to save perturbed images (optional)")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print("Running on", device)

    perturbation_levels = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
    noise_types = ['random_perturbation', 'salt_and_pepper', 'blur', 'gaussian', 'rectangles', 'swirl']
    image_paths = glob.glob(os.path.join(args.image_dir, '*.png'))

    # Initialize the results dictionary
    results = {img_path: {noise_type: {float(level): 0.0 for level in perturbation_levels} for noise_type in noise_types} for img_path in image_paths}

    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.mul(255))
    ])
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        for img_path in image_paths:
            original_image = utils.load_image(img_path)
            original_image = image_transform(original_image).unsqueeze(0).to(device)
            
            # Create temporary directories for original, style, and content images
            original_dir = os.path.join(tmpdirname, "original")
            style_dir = os.path.join(tmpdirname, "style")
            content_dir = os.path.join(tmpdirname, "content")
            os.makedirs(original_dir, exist_ok=True)
            os.makedirs(style_dir, exist_ok=True)
            os.makedirs(content_dir, exist_ok=True)

            # Save original image in all directories
            original_save_path = os.path.join(original_dir, "image.png")
            style_save_path = os.path.join(style_dir, "image.png")
            content_save_path = os.path.join(content_dir, "image.png")
            save_image(original_image.squeeze(0), original_save_path)
            shutil.copy(original_save_path, style_save_path)
            shutil.copy(original_save_path, content_save_path)

            for noise_type in noise_types:
                for perturbation in perturbation_levels:
                    if perturbation == 0.0:
                        # Compare original image with itself
                        avg_artfid = art_fid.compute_art_fid(original_dir, style_dir, content_dir, batch_size=1, device=device)
                    else:
                        if is_random_noise(noise_type):
                            total_artfid = 0.0
                            for run in range(args.runs):
                                perturbed_image = apply_noise(original_image.squeeze(0), noise_type, perturbation).unsqueeze(0)
                                
                                # Save perturbed image
                                perturbed_save_path = os.path.join(original_dir, "image.png")
                                save_image(perturbed_image.squeeze(0), perturbed_save_path)
                                
                                artfid_value = art_fid.compute_art_fid(original_dir, style_dir, content_dir, batch_size=1, device=device)
                                total_artfid += artfid_value
                            
                            avg_artfid = total_artfid / args.runs
                        else:
                            perturbed_image = apply_noise(original_image.squeeze(0), noise_type, perturbation).unsqueeze(0)
                            
                            # Save perturbed image
                            perturbed_save_path = os.path.join(original_dir, "image.png")
                            save_image(perturbed_image.squeeze(0), perturbed_save_path)
                            
                            avg_artfid = art_fid.compute_art_fid(original_dir, style_dir, content_dir, batch_size=1, device=device)

                    results[img_path][noise_type][float(perturbation)] = avg_artfid
                    print(f'Image: {os.path.basename(img_path)}, Noise: {noise_type}, Perturbation: {perturbation:.2f}, Avg ArtFID: {avg_artfid:.4f}')

                    # Save perturbed image if output directory is specified and it's the highest perturbation level
                    if args.output_dir and perturbation == max(perturbation_levels):
                        save_path = os.path.join(args.output_dir, f"{os.path.basename(img_path)[:-4]}_{noise_type}_perturbed.png")
                        save_image(perturbed_image.squeeze(0), save_path)

            # Clean up temporary directories
            shutil.rmtree(original_dir)
            shutil.rmtree(style_dir)
            shutil.rmtree(content_dir)

    # Export results to CSV
    with open('artfid_results.csv', 'w', newline='') as csvfile:
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
                    artfid_value = results[img_path][noise_type][level]
                    row.append(f'{artfid_value:.4f}')
                writer.writerow(row)
            
            # Add an empty line between tables
            writer.writerow([])
    
    print("Results exported to artfid_results.csv")
    if args.output_dir:
        print(f"Perturbed images saved in {args.output_dir}")

if __name__ == "__main__":
    main()