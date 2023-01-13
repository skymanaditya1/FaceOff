import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lpips import LPIPS
from models.discriminator import NLayerDiscriminator, weights_init

criterion = nn.L1Loss()

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

class VQLPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptual_loss = LPIPS().eval()

    def forward(self, targets, reconstructions):
        return self.perceptual_loss(targets.contiguous(), reconstructions.contiguous()).mean()

class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start,
                 disc_num_layers=3, disc_in_channels=3, 
                 disc_factor=1.0, disc_weight=0.8, use_actnorm=False,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)

        self.discriminator_iter_start = disc_start

        self.perceptual_loss = LPIPS().eval()
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

        self.perceptual_weight = 1.0

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, rec_loss, targets, reconstructions):
        p_loss = self.perceptual_loss(targets.contiguous(), reconstructions.contiguous())
        rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = torch.mean(rec_loss)

        return p_loss.mean(), nll_loss

    def second_forward(self, rec_loss, targets, reconstructions, optimizer_idx,
                global_step, perceptual_loss=False, last_layer=None):

        if perceptual_loss:
            p_loss = self.perceptual_loss(targets.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = 0

        nll_loss = torch.mean(rec_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            logits_fake = self.discriminator(reconstructions.contiguous())
            # g_loss = -torch.mean(logits_fake)
            g_loss = criterion(logits_fake, torch.ones_like(logits_fake, device=logits_fake.device))
            
            d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            return d_weight * disc_factor * g_loss, p_loss.mean(), nll_loss

        if optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(targets.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            # d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            loss_real = criterion(logits_real, torch.ones_like(logits_real, device=logits_real.device))
            loss_fake = criterion(logits_fake, torch.zeros_like(logits_fake, device=logits_fake.device))
            d_loss = disc_factor * (loss_real + loss_fake).mean()

            return d_loss, p_loss.mean(), nll_loss

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

class SiameseNetworkFaceSimilarity(nn.Module):
    def __init__(self):
        super(SiameseNetworkFaceSimilarity, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*256*256, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        x = x.unsqueeze(1)
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return F.pairwise_distance(output1, output2).mean()
