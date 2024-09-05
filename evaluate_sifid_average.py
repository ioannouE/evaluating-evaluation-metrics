import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import glob
import torch
from torchvision import transforms
from torchvision.utils import save_image
import tempfile
import csv
from skimage.util import random_noise
from skimage.transform import swirl
import utils
import shutil

# Add the SIFID folder to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sifid_dir = os.path.join(script_dir, 'SIFID')
sys.path.append(sifid_dir)

from sifid_score import calculate_sifid_given_paths

import numpy as np
np.random.seed(42)
from matplotlib.pyplot import imread


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

def apply_noise(image_tensor, noise_type, perturbation):
    # Convert to range [0, 255] before applying noise
    image_255 = image_tensor * 255

    if noise_type == 'none':
        return image_tensor

    elif noise_type == 'random_perturbation':
        perturbation_tensor = torch.randn(image_tensor.size()) * perturbation
        perturbed_image = image_255 + perturbation_tensor.to(image_tensor.device)
        perturbed_image = torch.clamp(perturbed_image, 0, 255)
        return perturbed_image # / 255

    elif noise_type == 'salt_and_pepper':
        noisy_image = random_noise(image_255.cpu().numpy(), mode='s&p', amount=0.1*perturbation, clip=True)
        return torch.tensor(noisy_image).to(image_tensor.device) # / 255

    elif noise_type == 'blur':
        kernel_sizes = {0.1:5, 0.25:11, 0.5: 17, 1.0:23, 2.0: 29, 3.0:35}
        kernel_size = kernel_sizes.get(perturbation, 5)
        gaussian_blur = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=perturbation*3)
        return gaussian_blur(image_255).to(image_tensor.device) # / 255

    elif noise_type == 'gaussian':
        noisy_image = random_noise(image_255.cpu().numpy(), mode='gaussian', mean=0, var=0.1*perturbation, clip=True)
        return torch.tensor(noisy_image).float().to(image_tensor.device)

    elif noise_type == 'rectangles':
        return utils.add_black_rectangles_tensor(image_255.clone().squeeze(), 10+int(10*perturbation), (10+int(30*perturbation), 10+int(30*perturbation))).unsqueeze(0) # / 255

    elif noise_type == 'swirl':
        image_np = image_255.squeeze().permute(1, 2, 0).cpu().numpy()
        swirled = swirl(image_np, radius=100 + (140 * perturbation), rotation=0, strength=1+int(perturbation), clip=True, center=(image_np.shape[1] / 2, image_np.shape[0] / 2), order=1)
        return torch.tensor(swirled).permute(2,0,1).unsqueeze(0).to(image_tensor.device) # / 255

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

def is_random_noise(noise_type):
    return noise_type in ['random_perturbation', 'gaussian', 'salt_and_pepper', 'rectangles', 'swirl' ]

def visualize_and_save_image(tensor_image, save_path):
    # Convert tensor to PIL Image for visualization
    image = transforms.ToPILImage()(tensor_image)
    
    # Save the image
    image.save(save_path)

def load_image_sifid(image_path):
    image = imread(image_path).astype(np.float32)
    image = image[:,:,:3]  # Ensure 3 channels (RGB)
    image = image.transpose((2, 0, 1))  # Reshape to (3, height, width)
    image /= 255.0  # Normalize to [0, 1]
    return image

def main():
    parser = argparse.ArgumentParser(description='Evaluate SIFID for perturbed images')
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

    image_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.mul(255))
    ])

    with tempfile.TemporaryDirectory() as tmpdirname:
        for img_path in image_paths:
            original_image = load_image_sifid(img_path)
            original_image = torch.from_numpy(original_image).unsqueeze(0).to(device)
            
            # Create temporary directories for original and perturbed images
            original_dir = os.path.join(tmpdirname, "original")
            perturbed_dir = os.path.join(tmpdirname, "perturbed")
            os.makedirs(original_dir, exist_ok=True)
            os.makedirs(perturbed_dir, exist_ok=True)

            # Save original image
            original_save_path = os.path.join(original_dir, "image.png")
            save_image(original_image.squeeze(0), original_save_path)

            for noise_type in noise_types:
                for perturbation in perturbation_levels:
                    if perturbation == 0.0:
                        # Compare original image with itself
                        shutil.copy(original_save_path, os.path.join(perturbed_dir, "image.png"))
                        sifid_values = calculate_sifid_given_paths(original_dir, perturbed_dir, 1, args.cuda, 64, 'png')
                        avg_sifid = np.mean(sifid_values) if len(sifid_values) > 0 else 0.0
                    else:
                        if is_random_noise(noise_type):
                            total_sifid = 0.0
                            for run in range(args.runs):
                                perturbed_image = apply_noise(original_image.squeeze(0), noise_type, perturbation).unsqueeze(0)
                                
                                # Save perturbed image
                                perturbed_save_path = os.path.join(perturbed_dir, "image.png")
                                save_image(perturbed_image.squeeze(0), perturbed_save_path)
                                
                                sifid_values = calculate_sifid_given_paths(original_dir, perturbed_dir, 1, args.cuda, 64, 'png')
                                total_sifid += np.mean(sifid_values) if len(sifid_values) > 0 else 0.0
                            
                            avg_sifid = total_sifid / args.runs
                        else:
                            perturbed_image = apply_noise(original_image.squeeze(0), noise_type, perturbation).unsqueeze(0)
                            
                            # Save perturbed image
                            perturbed_save_path = os.path.join(perturbed_dir, "image.png")
                            save_image(perturbed_image.squeeze(0), perturbed_save_path)
                            
                            sifid_values = calculate_sifid_given_paths(original_dir, perturbed_dir, 1, args.cuda, 64, 'png')
                            avg_sifid = np.mean(sifid_values) if len(sifid_values) > 0 else 0.0

                    results[img_path][noise_type][float(perturbation)] = avg_sifid
                    print(f'Image: {os.path.basename(img_path)}, Noise: {noise_type}, Perturbation: {perturbation:.2f}, Avg SIFID: {avg_sifid:.4f}')

                    # Save perturbed image if output directory is specified and it's the highest perturbation level
                    if args.output_dir and perturbation == max(perturbation_levels):
                        save_path = os.path.join(args.output_dir, f"{os.path.basename(img_path)[:-4]}_{noise_type}_perturbed.png")
                        save_image(perturbed_image.squeeze(0), save_path)

            # Clean up temporary directories
            shutil.rmtree(original_dir)
            shutil.rmtree(perturbed_dir)

    # Export results to CSV
    with open('sifid_results.csv', 'w', newline='') as csvfile:
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
                    sifid_value = results[img_path][noise_type][float(level)]
                    row.append(f'{sifid_value:.4f}')
                writer.writerow(row)
            
            # Add an empty line between tables
            writer.writerow([])
    
    print("Results exported to sifid_results.csv")
    if args.output_dir:
        print(f"Perturbed images saved in {args.output_dir}")

if __name__ == "__main__":
    main()