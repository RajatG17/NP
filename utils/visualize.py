import torchvision.utils as vutils


def save_image(tensor, filename):
    vutils.save_image(tensor, filename, normalize=True)