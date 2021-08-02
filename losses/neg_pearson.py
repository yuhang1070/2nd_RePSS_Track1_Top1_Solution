import torch
import torch.nn as nn

"""
Code from:
https://github.com/terbed/Deep-rPPG/blob/master/src/errfuncs.py
"""

tr = torch


class Neg_Pearson(nn.Module):
    def __init__(self):
        super(Neg_Pearson, self).__init__()

    def forward(self, x, y):
        if len(x.size()) == 1:
            x = tr.unsqueeze(x, 0)
            y = tr.unsqueeze(y, 0)
        T = x.shape[1]
        p_coefficient = tr.sub(T * tr.sum(tr.mul(x, y), 1), tr.mul(tr.sum(x, 1), tr.sum(y, 1)))
        norm = tr.sqrt((T * tr.sum(x ** 2, 1) - tr.sum(x, 1) ** 2) * (T * tr.sum(y ** 2, 1) - tr.sum(y, 1) ** 2))
        p_coefficient = tr.div(p_coefficient, norm)
        losses = tr.tensor(1.) - p_coefficient
        tot_loss = tr.mean(losses)
        return tot_loss


if __name__ == '__main__':
    pred = torch.tensor([[1, 2, 3, 4], [4, 3, 1, 0]])
    gt = torch.tensor([[1.1, 2.3, 2.8, 4.3], [4.2, 3.2, 2.0, 1.5]])

    loss1 = Neg_Pearson()

    print(loss1(pred, gt))  # tensor(0.0120)
    print(loss1(gt, pred))  # tensor(0.0120)
