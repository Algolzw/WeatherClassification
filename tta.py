import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

# mean = [0.5, 0.5, 0.5]
# std = [0.5, 0.5, 0.5]

mean = [0.507, 0.522, 0.500]
std = [0.213, 0.207, 0.212]

def visualizationImage(imgs):
    for index, img in enumerate(imgs):
        plt.subplot(1, 4, index + 1)
        plt.imshow(img)
        plt.axis('off')

class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        # std_inv = 1.0 / std
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

def TensorToPILs(inputs):
    # unNorm=NormalizeInverse(mean=mean, std=std)
    # imgs = [F.to_pil_image(unNorm(inputs[i])) for i in range(inputs.shape[0])]
    imgs = [F.to_pil_image(inputs[i]) for i in range(inputs.shape[0])]
    return imgs

def PILsToTensor(imgs):
    Norm = transforms.Normalize(mean=mean,std=std)
    tensors = [Norm(F.to_tensor(img)) for img in imgs]
    # tensors = [F.to_tensor(img) for img in imgs]
    return torch.stack(tensors)

class NoneAug():
    def __call__(self, imgs):
        return imgs

class Resize():
    def __init__(self, size):
        self.size = size

    def __call__(self, imgs):
        return [F.resize(img=img, size=self.size) for img in imgs]

class Hflip():
    def __call__(self,imgs):
        return [F.hflip(img=img) for img in imgs]

class Vflip():
    def __call__(self, imgs):
        return [F.vflip(img=img) for img in imgs]

class Rotate():
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, imgs):
        return [F.rotate(img=img, angle=self.angle) for img in imgs]

class Grayscale():
    def __init__(self, output_channels=1):
        self.output_channels = output_channels
    def __call__(self, imgs):
        return [F.to_grayscale(img=img, num_output_channels=self.output_channels) for img in imgs]

class Adjustbright():
    def __init__(self, bright_factor):
        self.bright_factor = bright_factor
    def __call__(self, imgs):
        return [F.adjust_brightness(img=img, brightness_factor=self.bright_factor) for img in imgs]

class Adjustcontrast():
    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor
    def __call__(self, imgs):
        return [F.adjust_contrast(img=img, contrast_factor=self.contrast_factor) for img in imgs]

class Adjustsaturation():
    def __init__(self, saturation_factor):
        self.saturation_factor = saturation_factor
    def __call__(self, imgs):
        return [F.adjust_saturation(img=img, saturation_factor=self.saturation_factor) for img in imgs]

class Adjustgamma():
    def __init__(self, gamma, gain=1):
        self.gamma = gamma
        self.gain = gain
    def __call__(self, imgs):
        return [F.adjust_gamma(img=img, gamma=self.gamma, gain=self.gain) for img in imgs]


