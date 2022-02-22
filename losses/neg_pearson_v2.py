import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Code from:
https://github.com/terbed/Deep-rPPG/blob/master/src/errfuncs.py
"""

tr = torch


class Neg_Pearson(nn.Module):
    def __init__(self):
        super(Neg_Pearson, self).__init__()

    def forward(self, x, y):
        if x.size() != y.size():
            raise Exception('`x` and `y` MUST have same size!!!')
        elif len(x.size()) != 2:
            raise Exception('`len(x.size())` MUST equal 2!!!')
        # if len(x.size()) == 1:
        #     x = tr.unsqueeze(x, 0)
        #     y = tr.unsqueeze(y, 0)
        # T = x.shape[1]
        # p_coefficient = tr.sub(T * tr.sum(tr.mul(x, y), 1), tr.mul(tr.sum(x, 1), tr.sum(y, 1)))
        # norm = tr.sqrt((T * tr.sum(x ** 2, 1) - tr.sum(x, 1) ** 2) * (T * tr.sum(y ** 2, 1) - tr.sum(y, 1) ** 2))
        # p_coefficient = tr.div(p_coefficient, norm)
        # losses = tr.tensor(1.) - p_coefficient
        # tot_loss = tr.mean(losses)
        tot_loss = 1 - torch.mean(torch.cosine_similarity(x, y, dim=1))
        return tot_loss


if __name__ == '__main__':
    pred = torch.tensor([[1, 2, 2., 4], [4, 3, 1, 1], [4., 3.2, 1.0, 1.2]], dtype=torch.float32)
    gt = torch.tensor([[-1, -2, -2., -4], [4, 3, 1, 1], [4., 3.2, 1.0, 1.2]], dtype=torch.float32)

    # print(pred.mean(dim=1, keepdim=True))
    # pred = (pred - pred.mean(dim=1, keepdim=True)) / pred.std(dim=1, keepdim=True)
    # gt = (gt - gt.mean(dim=1, keepdim=True)) / gt.std(dim=1, keepdim=True)
    # print(pred.mean(dim=1, keepdim=True))
    print(pred.shape)
    pred = pred.reshape([3, 4])
    gt = gt.reshape([3, 4])
    loss1 = Neg_Pearson()

    print(loss1(pred, gt))  # tensor(0.0120)
    print(loss1(gt, pred))  # tensor(0.0120)
    print(loss1(pred, gt).shape)

    print(1 - torch.mean(torch.cosine_similarity(pred, gt, dim=1)))
