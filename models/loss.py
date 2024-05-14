import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

class GeneratorLoss(GANLoss):
    def __init__(self):
        super(GeneratorLoss, self).__init__()

    def __call__(self, fake_output):
        loss = super().__call__(fake_output, True)
        return loss

class DiscriminatorLoss(GANLoss):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def __call__(self, real_output, fake_output):
        real_loss = super().__call__(real_output, True)
        fake_loss = super().__call__(fake_output, False)
        loss = (real_loss + fake_loss) / 2
        return loss