import torchvision.transforms as transforms
import torch.nn as nn
import torch

class RGBPreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 网页7/8
            transforms.ColorJitter(brightness=0.3, contrast=0.3),  # 网页4
            transforms.GaussianBlur(kernel_size=5),  # 网页12
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, img):
        return self.transform(img)