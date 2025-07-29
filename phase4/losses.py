import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pytorch_msssim import ms_ssim

# Sobel kernels (for EdgeLoss)
sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
sobel_y = sobel_x.transpose(2,3)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16].eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg
        self.l1  = nn.L1Loss()

    def forward(self, pred, gt):
        return self.l1(self.vgg(pred), self.vgg(gt))

class EdgeLoss(nn.Module):
    def forward(self, pred, gt):
        # convert to gray
        p_gray = (0.2989*pred[:,0] + 0.5870*pred[:,1] + 0.1140*pred[:,2]).unsqueeze(1)
        g_gray = (0.2989*gt[:,0]   + 0.5870*gt[:,1]   + 0.1140*gt[:,2]).unsqueeze(1)
        ex_p = F.conv2d(p_gray, sobel_x.to(pred.device), padding=1)
        ex_g = F.conv2d(g_gray, sobel_x.to(gt.device),   padding=1)
        ey_p = F.conv2d(p_gray, sobel_y.to(pred.device), padding=1)
        ey_g = F.conv2d(g_gray, sobel_y.to(gt.device),   padding=1)
        return (ex_p - ex_g).abs().mean() + (ey_p - ey_g).abs().mean()

def combined_loss(coarse, final, gt, perc_loss, edge_loss,
                  λ_l1=1.0, λ_ssim=1.0, λ_perc=0.1, λ_edge=0.1):
    # deep supervision on coarse
    l1_1   = F.l1_loss(coarse, gt)
    ssim_1 = 1 - ms_ssim(coarse, gt, data_range=1.0)
    loss1  = λ_l1 * l1_1 + λ_ssim * ssim_1

    # main loss on final
    l1_2   = F.l1_loss(final, gt)
    ssim_2 = 1 - ms_ssim(final, gt, data_range=1.0)
    perc   = perc_loss(final, gt)
    edge   = edge_loss(final, gt)
    loss2  = λ_l1 * l1_2 + λ_ssim * ssim_2 + λ_perc * perc + λ_edge * edge

    return loss2 + 0.5 * loss1
