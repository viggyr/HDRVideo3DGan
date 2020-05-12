import torch.nn.functional as F
import torch

def cross_entropy_adversarial_loss(fake_hdr):
    ones = torch.ones(fake_hdr.shape[0]).cuda()
    return nn.BCEWithLogitsLoss(fake_hdr, ones)

def l1_loss(output, target):
    return torch.nn.L1Loss(output, target)

def cross_entropy_loss(real_hdr, fake_hdr):
    ones = torch.ones(real_hdr.shape[0]).cuda()
    zeros = torch.zeros(real_hdr.shape[0]).cuda()
    return nn.BCEWithLogitsLoss(fake_hdr, zeros) + nn.BCEWithLogitsLoss(pred_real, ones))/2
