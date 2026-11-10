"""Loss: BCE + soft Dice on the soft semantic target; auxiliary spine (BCE+Dice) and boundary (BCE) heads."""
import torch
import torch.nn.functional as F


def soft_dice(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    return 1 - (2 * inter + eps) / (p.sum() + target.sum() + eps)


class FilamentLoss(torch.nn.Module):
    def __init__(self, w_spine=0.5, w_bnd=0.3):
        super().__init__()
        self.w_spine, self.w_bnd = w_spine, w_bnd

    def forward(self, logits, y):
        logits = logits.float()
        sem = F.binary_cross_entropy_with_logits(logits[:, 0], y[:, 0]) + soft_dice(logits[:, 0], y[:, 0])
        spine = F.binary_cross_entropy_with_logits(logits[:, 1], y[:, 1]) + soft_dice(logits[:, 1], y[:, 1])
        bnd = F.binary_cross_entropy_with_logits(logits[:, 2], y[:, 2])
        total = sem + self.w_spine * spine + self.w_bnd * bnd
        return total, {"sem": sem.item(), "spine": spine.item(), "bnd": bnd.item()}
