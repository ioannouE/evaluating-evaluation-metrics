''' 
based on utils of PyTorch implementation of Johnson et al 
https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
'''
import torch
from PIL import Image
import random
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from skimage.util import random_noise
from skimage.transform import swirl, warp


def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int (img.size[1] / scale)), Image.ANTIALIAS)
    
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)

    return gram


def normalize_batch(batch):
    # normalises using imagenet and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.255]).view(-1, 1, 1)
    batch = batch.div_(255.0)

    return (batch - mean) / std


def visualize_image(tensor_image):
    # Convert tensor to PIL Image for visualization
    image = transforms.ToPILImage()(tensor_image)

    # Display the image
    plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
    plt.imshow(image)
    plt.axis('off')  # Remove axis labels
    # plt.show()
    plt.savefig('viz.png')

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


def get_subdirectories(directory_path):
    subdirectories = []
    for entry in os.scandir(directory_path):
        if entry.is_dir():
            subdirectories.append(entry.name)
    return subdirectories

