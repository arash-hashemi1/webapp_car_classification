# Description: This file contains the transform function for the dataset.
import torch
# import torchvision.transforms as v2
from torchvision import transforms


def transform_image():
  mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
  custom_to_dtype = transforms.Lambda(lambda x: transforms.ToTensor()(x).to(torch.float32))
  image_transforms = transforms.Compose([
    # v2.ToPILImage(),
    transforms.Resize((400,400), antialias=True),
    transforms.RandomCrop(350),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    custom_to_dtype,
    transforms.Normalize(mean, std, inplace=True)
    ])

  return image_transforms