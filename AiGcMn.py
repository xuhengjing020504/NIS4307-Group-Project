import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from cDCGAN import Generator

class AiGcMn:
    def __init__(self, generator_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator().to(self.device)
        self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
        self.generator.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
        ])
    
    def generate_images(self, num_images, noise_dim=100, class_labels=[0]):
        assert len(class_labels) == num_images, "Number of class labels should match the number of images"
        
        noise = torch.randn(num_images, noise_dim).to(self.device)
        class_vector = torch.zeros(num_images, 10).to(self.device)
        for i, label in enumerate(class_labels):
            class_vector[i, label] = 1
        
        with torch.no_grad():
            generated_images = self.generator(noise, class_vector)
        generated_images = (generated_images + 1) / 2  # Scale pixel values from [-1, 1] to [0, 1]
        generated_images = generated_images.cpu()  # Move images to CPU if necessary
        generated_images = torch.squeeze(generated_images)  # Remove single-dimensional dimensions
        
        # Convert generated images to PIL Images
        pil_images = [self.transform(img) for img in generated_images]
        
        return pil_images
