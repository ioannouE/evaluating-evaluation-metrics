import numpy as np
import torch
from PIL import Image


def compute_accuracy(pred, target):
    # remove entries where target is -1
    idcs = target != -1
    if torch.sum(idcs) == 0:
        return -1

    target = target[idcs]
    pred = pred[idcs]

    _, pred_label = torch.max(pred, dim=-1)
    acc = torch.sum(pred_label == target) / pred_label.shape[0]
    return acc


def compute_accuracy(pred, target):
    # remove entries where target is -1
    idcs = target != -1
    if torch.sum(idcs) == 0:
        return -1, -1

    target = target[idcs]
    pred = pred[idcs]

    _, pred_label = torch.max(pred, dim=-1)
    acc = torch.sum(pred_label == target) / pred_label.shape[0]

    # top5
    pred_label_top5 = torch.topk(pred, dim=1, k=5).indices
    target_top5 = torch.repeat_interleave(target.unsqueeze(dim=1), 5, dim=1)
    acc_top5 = torch.sum(torch.sum(pred_label_top5 == target_top5, dim=1) >= 1) / pred_label.shape[0]
    return acc, acc_top5


def tensor_to_numpy(x):
    if len(x.shape) == 4:
        return np.transpose(x.data.cpu(), axes=(0, 2, 3, 1)) 
    if len(x.shape) == 5:
        return np.transpose(x.data.cpu(), axes=(0, 1, 3, 4, 2)) 


def post_process_img(x):
    x = tensor_to_numpy(x)
    return (x + 1) / 2.0


def img_to_tensor(path, expand=True, transforms=None, dtype='float32'):
    img = Image.open(path)
    x = np.array(img, dtype=dtype)
    if transforms is not None:
        x = transforms(image=x)['image']
    if expand:
        x = np.expand_dims(x, axis=0)
        x = np.transpose(x, axes=(0, 3, 1, 2))
    else:
        x = np.transpose(x, axes=(2, 0, 1))
    x /= 255.0
    x = 2 * x - 1
    t = torch.from_numpy(x)
    return t


def tensor_to_imgs(t):
    imgs = []
    num_imgs = 1 if len(t.shape) < 4 else t.shape[0]
    for i in range(num_imgs):
        x = t[i].data.cpu().numpy()
        x = np.transpose(x, axes=(1, 2, 0))
        x = (x + 1) / 2.0
        x *= 255.0
        img = Image.fromarray(x.astype('uint8'))
        imgs.append(img)
    return imgs if num_imgs > 1 else imgs[0]


