import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import inception_v3
import numpy as np
from scipy.linalg import sqrtm


class FIDScore:
    def __init__(self, device='cuda'):
        self.device = device
        self.inception_model = inception_v3(pretrained=True, transform_input=False, aux_logits=True).to(device)
        self.inception_model.eval()
        self.to_pil = transforms.ToPILImage()
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def calculate_activations(self, images):
        activations = []
        with torch.no_grad():
            for image in images:
                if isinstance(image, torch.Tensor):
                    if image.dim() == 3:
                        image = self.to_pil(image)
                    elif image.dim() == 4 and image.shape[1] == 3:
                        image = self.to_pil(image[0])  # Take the first batch element
                elif not isinstance(image, Image):
                    raise ValueError("Unsupported image type. Must be PIL Image or 3D/4D Tensor.")

                image = self.transform(image).unsqueeze(0).to(self.device)
                pred = self.inception_model(image)[0]
                if len(pred.shape) == 2:
                    pred = pred.flatten(1)
                activations.append(pred.cpu().numpy().squeeze())

        return np.array(activations)

    def calculate_fid(self, real_images, fake_images):
        real_activations = self.calculate_activations(real_images)
        fake_activations = self.calculate_activations(fake_images)

        mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
        mu_fake, sigma_fake = np.mean(fake_activations, axis=0), np.cov(fake_activations, rowvar=False)

        # Calculate FID score
        eps = 1e-6
        cov_sqrt = sqrtm(sigma_real.dot(sigma_fake) + eps)
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real
        fid_score = np.sum((mu_real - mu_fake) ** 2) + np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)

        return fid_score
