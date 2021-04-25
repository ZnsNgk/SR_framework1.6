import torch
import math

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, eps=1e-6):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = eps

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

loss_func_list = {
    "L1": torch.nn.L1Loss,
    "L2": torch.nn.MSELoss,
    "L1_Charbonnier": L1_Charbonnier_loss,
}

def get_loss_func(name, *args):
    return loss_func_list[name](*args)

def loss_PSNR_L1(loss):
    return 10. * math.log10(65025. / (loss**2))

def loss_PSNR_L1_norm(loss):
    return 10. * math.log10(1. / (loss**2))

def loss_PSNR_L2(loss):
    return 10. * math.log10(65025. / loss)

def loss_PSNR_L2_norm(loss):
    return 10. * math.log10(1. / loss)

cal_list = {
    "L1": loss_PSNR_L1,
    "L2": loss_PSNR_L2,
    "L1_Charbonnier": loss_PSNR_L1,
}

cal_list_norm = {
    "L1": loss_PSNR_L1_norm,
    "L2": loss_PSNR_L2_norm,
    "L1_Charbonnier": loss_PSNR_L1_norm,
}

def get_loss_PSNR(name, normal):
    if normal:
        return cal_list_norm[name]
    else:
        return cal_list[name]