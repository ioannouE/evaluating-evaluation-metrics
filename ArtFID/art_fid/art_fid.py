import argparse
import glob
import numpy as np
import os
from PIL import Image
from scipy import linalg
from sklearn.linear_model import LinearRegression
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from tqdm import tqdm

import utils
import inception
import image_metrics
from skimage.util import random_noise
from torchvision import transforms
import matplotlib.pyplot as plt
import utils
from skimage.transform import swirl

ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
CKPT_URL = 'https://huggingface.co/matthias-wright/art_inception/resolve/main/art_inception.pth'


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def visualize_image(tensor_image):
    # Convert tensor to PIL Image for visualization
    image = transforms.ToPILImage()(tensor_image)

    # Display the image
    plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
    plt.imshow(image)
    plt.axis('off')  # Remove axis labels
    plt.show()


def get_activations(files, model, batch_size=50, device='cpu', num_workers=1, perturbation=0, noise=False):
    """Computes the activations of for all images.

    Args:
        files (list): List of image file paths.
        model (torch.nn.Module): Model for computing activations.
        batch_size (int): Batch size for computing activations.
        device (torch.device): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (): Activations of the images, shape [num_images, 2048].
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    transform = Compose([Resize((512, 512)), ToTensor()])

    # dataset = ImagePathDataset(files, transforms=ToTensor())
    dataset = ImagePathDataset(files, transforms=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), 2048))
    start_idx = 0
    pbar = tqdm(total=len(files))
    for batch in dataloader:
        batch = batch.to(device)

        if (noise):
            # perturbation_strength = perturbation
            # perturbation = torch.randn(batch.size()) * perturbation_strength
            # perturbed_image = batch + perturbation.to(device)
            # perturbed_image = torch.clamp(perturbed_image, 0, 1)
            # batch = perturbed_image


            # gaussian blur
            # kernel_sizes = {0.1:5, 0.25:11, 0.5: 17, 1.0:23, 2.0: 29, 3.0:35}
            # gaussian_blur = transforms.GaussianBlur(kernel_size=(kernel_sizes[perturbation],kernel_sizes[perturbation]), sigma=perturbation*3)
            # batch = gaussian_blur(batch)

            # visualize_image(batch[0])

            # salt and pepper
            # batch = batch.cpu().numpy()
            # batch = torch.tensor(random_noise(batch, mode='s&p', amount=0.1*perturbation, clip=True))
            # batch = batch.to(device)

            batch_size = batch.size(0)

            # gaussian noise
            # batch = batch.cpu().numpy()
            # batch = torch.tensor(random_noise(batch, mode='gaussian', mean=0, var=0.1*perturbation, clip=True))
            # batch = batch.float().to(device)

            # implanted black rectangles
            # for i in range(batch_size):
            #     batch[i] = utils.add_black_rectangles_tensor(batch[i].clone().squeeze(), 10+int(10*perturbation), (10+int(30*perturbation), 10+int(30*perturbation)))

            # swirl
            # batch is of dimensions [20, 3, 512, 512]
            batch_size = batch.size(0)

            for i in range(batch_size):
                image_np = batch[i].squeeze().permute(1, 2, 0).cpu().numpy()
                swirled = swirl(image_np, radius=100 + (140 * perturbation), rotation=0, strength=1 + int(perturbation), clip=True, center=(image_np.shape[1] / 2, image_np.shape[0] / 2), order=1)
                batch[i] = torch.tensor(swirled).permute(2, 0, 1)

            # utils.visualize_image(batch[0])

        with torch.no_grad():
            features = model(batch, return_features=True)

        features = features.cpu().numpy()
        pred_arr[start_idx:start_idx + features.shape[0]] = features
        start_idx = start_idx + features.shape[0]

        pbar.update(batch.shape[0])

    pbar.close()
    return pred_arr


def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    
    Args:
        mu1 (np.ndarray): Sample mean of activations of stylized images.
        mu2 (np.ndarray): Sample mean of activations of style images.
        sigma1 (np.ndarray): Covariance matrix of activations of stylized images.
        sigma2 (np.ndarray): Covariance matrix of activations of style images.
        eps (float): Epsilon for numerical stability.

    Returns:
        (float) FID value.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def compute_activation_statistics(files, model, batch_size=50, device='cpu', num_workers=1):
    """Computes the activation statistics used by the FID.
    
    Args:
        files (list): List of image file paths.
        model (torch.nn.Module): Model for computing activations.
        batch_size (int): Batch size for computing activations.
        device (torch.device): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (np.ndarray, np.ndarray): mean of activations, covariance of activations
        
    """
    act = get_activations(files, model, batch_size, device, num_workers, perturbation=0, noise=False)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_image_paths(path, sort=False):
    """Returns the paths of the images in the specified directory, filtered by allowed file extensions.

    Args:
        path (str): Path to image directory.
        sort (bool): Sort paths alphanumerically.

    Returns:
        (list): List of image paths with allowed file extensions.

    """
    paths = []
    for extension in ALLOWED_IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(path, f'*.{extension}')))
    if sort:
        paths.sort()
    return paths


def compute_fid(path_to_stylized, path_to_style, batch_size, device, num_workers=1, perturbation=0, noise=False):
    """Computes the FID for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        batch_size (int): Batch size for computing activations.
        device (str): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) FID value.
    """
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    ckpt_file = utils.download(CKPT_URL)
    ckpt = torch.load(ckpt_file, map_location=device)
    
    model = inception.Inception3().to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    stylized_image_paths = get_image_paths(path_to_stylized)
    style_image_paths = get_image_paths(path_to_style)

    mu1, sigma1 = compute_activation_statistics(stylized_image_paths, model, batch_size, device, num_workers)
    mu2, sigma2 = compute_activation_statistics(style_image_paths, model, batch_size, device, num_workers)
    
    fid_value = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value


def compute_fid_infinity(path_to_stylized, path_to_style, batch_size, device, num_points=15, num_workers=1, perturbation=0., noise=False):
    """Computes the FID infinity for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        batch_size (int): Batch size for computing activations.
        device (str): Device for commputing activations.
        num_points (int): Number of FID_N we evaluate to fit a line.
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) FID infinity value.
    """
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    ckpt_file = utils.download(CKPT_URL)
    ckpt = torch.load(ckpt_file, map_location=device)
    
    model = inception.Inception3().to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    stylized_image_paths = get_image_paths(path_to_stylized)
    style_image_paths = get_image_paths(path_to_style)

    assert len(stylized_image_paths) == len(style_image_paths), \
           'Number of stylized images and number of style images must be equal.'

    activations_stylized = get_activations(stylized_image_paths, model, batch_size, device, num_workers, perturbation=perturbation, noise=noise)
    activations_style = get_activations(style_image_paths, model, batch_size, device, num_workers, perturbation=perturbation, noise=False)
    activation_idcs = np.arange(activations_stylized.shape[0])

    fids = []
    
    fid_batches = np.linspace(start=5000, stop=len(stylized_image_paths), num=num_points).astype('int32')
    
    for fid_batch_size in fid_batches:
        np.random.shuffle(activation_idcs)
        idcs = activation_idcs[:fid_batch_size]
        
        act_style_batch = activations_style[idcs]
        act_stylized_batch = activations_stylized[idcs]

        mu_style, sigma_style = np.mean(act_style_batch, axis=0), np.cov(act_style_batch, rowvar=False)
        mu_stylized, sigma_stylized = np.mean(act_stylized_batch, axis=0), np.cov(act_stylized_batch, rowvar=False)
        
        fid_value = compute_frechet_distance(mu_style, sigma_style, mu_stylized, sigma_stylized)
        fids.append(fid_value)

    fids = np.array(fids).reshape(-1, 1)
    reg = LinearRegression().fit(1 / fid_batches.reshape(-1, 1), fids)
    fid_infinity = reg.predict(np.array([[0]]))[0,0]

    return fid_infinity


def compute_content_distance(path_to_stylized, path_to_content, batch_size, content_metric='lpips', device='cuda', num_workers=1, perturbation=0, noise=False):
    """Computes the distance for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        batch_size (int): Batch size for computing activations.
        content_metric (str): Metric to use for content distance. Choices: 'lpips', 'vgg', 'alexnet'
        device (str): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) FID value.
    """
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    # Sort paths in order to match up the stylized images with the corresponding content image
    stylized_image_paths = get_image_paths(path_to_stylized, sort=True)
    content_image_paths = get_image_paths(path_to_content, sort=True)

    assert len(stylized_image_paths) == len(content_image_paths), \
           'Number of stylized images and number of content images must be equal.'

    if content_metric == 'clip':
        content_transforms = Compose([Resize((224,224), interpolation=InterpolationMode.BICUBIC),
                                      CenterCrop(224),
                                      lambda img : img.convert('RGB'),
                                      ToTensor(),
                                      Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    else:
        content_transforms = ToTensor()
        content_transforms = Compose([Resize((224, 224)), ToTensor()])
    
    dataset_stylized = ImagePathDataset(stylized_image_paths, transforms=content_transforms)
    dataloader_stylized = torch.utils.data.DataLoader(dataset_stylized,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=num_workers)

    dataset_content = ImagePathDataset(content_image_paths, transforms=content_transforms)
    dataloader_content = torch.utils.data.DataLoader(dataset_content,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=num_workers)
    
    if content_metric == 'vgg' or content_metric == 'alexnet':
        metric = image_metrics.Metric(content_metric).to(device)
    elif content_metric == 'lpips':
        metric = image_metrics.LPIPS().to(device)
    else:
        raise ValueError(f'Invalid content metric: {content_metric}')

    dist_sum = 0.0
    N = 0
    pbar = tqdm(total=len(stylized_image_paths))
    for batch_stylized, batch_content in zip(dataloader_stylized, dataloader_content):
        with torch.no_grad():
            if (noise):
                # perturbation_strength = perturbation
                # perturbation = torch.randn(batch_stylized.size()) * perturbation_strength
                # perturbed_image = batch_stylized + perturbation
                # perturbed_image = torch.clamp(perturbed_image, 0, 1)
                # batch_stylized = perturbed_image

                # gaussian blur
                # gaussian_blur = transforms.GaussianBlur(kernel_size=(5,5), sigma=perturbation)
                # kernel_sizes = {0.1:5, 0.25:11, 0.5: 17, 1.0:23, 2.0: 29, 3.0:35}
                # gaussian_blur = transforms.GaussianBlur(kernel_size=(kernel_sizes[perturbation],kernel_sizes[perturbation]), sigma=perturbation*3)
                # batch_stylized = gaussian_blur(batch_stylized)

                # salt and pepper
                # batch_stylized = torch.tensor(random_noise(batch_stylized, mode='s&p', amount=0.1*perturbation, clip=True))

                # batch is of dimensions [20, 3, 512, 512]
                batch_size = batch_stylized.size(0)

                # gaussian noise
                # batch_stylized = batch_stylized.cpu().numpy()
                # batch_stylized = torch.tensor(random_noise(batch_stylized, mode='gaussian', mean=0, var=0.1*perturbation, clip=True))
                # batch_stylized = batch_stylized.float().to(device)

                # print(batch_stylized.size())
                
                # implanted black rectangles
                # for i in range(batch_size):
                #     batch_stylized[i] = utils.add_black_rectangles_tensor(batch_stylized[i].clone().squeeze(), 10+int(10*perturbation), (10+int(30*perturbation), 10+int(30*perturbation)))


                # swirl
                for i in range(batch_size):
                    image_np = batch_stylized[i].squeeze().permute(1, 2, 0).cpu().numpy()
                    swirled = swirl(image_np, radius=100 + (140 * perturbation), rotation=0, strength=1 + int(perturbation), clip=True, center=(image_np.shape[1] / 2, image_np.shape[0] / 2), order=1)
                    batch_stylized[i] = torch.tensor(swirled).permute(2, 0, 1)

                # utils.visualize_image(batch_stylized[0])

            batch_dist = metric(batch_stylized.to(device), batch_content.to(device))
            N += batch_dist.shape[0]
            dist_sum += torch.sum(batch_dist)

        pbar.update(batch_stylized.shape[0])

    pbar.close()

    return dist_sum / N


def compute_art_fid(path_to_stylized, path_to_style, path_to_content, batch_size, device, mode='art_fid_inf', content_metric='lpips', num_workers=1, perturbation=0, noise=False):
    """Computes the FID for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        path_to_content (str): Path to the content images.
        batch_size (int): Batch size for computing activations.
        device (str): Device for commputing activations.
        content_metric (str): Metric to use for content distance. Choices: 'lpips', 'vgg', 'alexnet'
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) ArtFID value.
    """
    print('Compute FID value...')
    if mode == 'art_fid_inf':
        fid_value = compute_fid_infinity(path_to_stylized, path_to_style, batch_size, device, num_workers, perturbation=perturbation, noise=noise)
    else:
        fid_value = compute_fid(path_to_stylized, path_to_style, batch_size, device, num_workers, perturbation=perturbation, noise=noise)
    
    print('Compute content distance...')
    content_dist = compute_content_distance(path_to_stylized, path_to_content, batch_size, content_metric, device, num_workers, perturbation=perturbation, noise=noise)

    art_fid_value = (content_dist + 1) * (fid_value + 1)
    return art_fid_value.item()

