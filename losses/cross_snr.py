import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

"""
Code from:
https://github.com/nxsEdson/CVD-Physiological-Measurement/blob/master/utils/loss/loss_SNR.py
"""


class Cross_SNR_Loss(nn.Module):
    def __init__(self, clip_length=300, delta=3, loss_type=7, device=None):
        super(Cross_SNR_Loss, self).__init__()

        self.clip_length = clip_length
        self.time_length = 300
        self.delta = delta
        self.low_bound = 40
        self.high_bound = 250

        self.loss_type = loss_type
        self.device = device

        self.pi = 3.14159265

        self.bpm_range = torch.arange(self.low_bound, self.high_bound, dtype=torch.float32, device=self.device) / 60.0

        self.two_pi_n = Variable(
            2 * self.pi * torch.arange(0, self.time_length, dtype=torch.float32, device=self.device))
        self.hanning = Variable(torch.from_numpy(np.hanning(self.time_length)).type(torch.FloatTensor),
                                requires_grad=True).view(1, -1).to(self.device)

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, wave, gt_bpm, fps, ):  # all variable operation
        hr = torch.mul(gt_bpm, fps)
        hr = hr * 60 / self.clip_length
        hr[hr.ge(self.high_bound)] = self.high_bound - 1
        hr[hr.le(self.low_bound)] = self.low_bound

        batch_size = wave.shape[0]

        self.bpm_range = self.bpm_range.repeat(batch_size, 1)
        fps = fps.view(batch_size, 1)
        f_t = self.bpm_range / fps
        preds = wave * self.hanning

        preds = preds.view(batch_size, 1, -1)
        f_t = f_t.view(batch_size, -1, 1)

        tmp = self.two_pi_n.repeat(batch_size, 1)
        tmp = tmp.view(batch_size, 1, -1)

        complex_absolute = torch.sum(preds * torch.sin(f_t * tmp), dim=-1) ** 2 \
                           + torch.sum(preds * torch.cos(f_t * tmp), dim=-1) ** 2

        # print('complex_absolute: {}'.format(complex_absolute))

        target = hr - self.low_bound
        target = target.type(torch.long).view(batch_size)

        whole_max_val, whole_max_idx = complex_absolute.max(1)
        whole_max_idx = whole_max_idx + self.low_bound

        if self.loss_type == 1:
            loss = self.cross_entropy(complex_absolute, target)

        elif self.loss_type == 7:
            norm_t = (torch.ones(batch_size).cuda() / torch.sum(complex_absolute, dim=1))
            norm_t = norm_t.view(-1, 1)
            complex_absolute = complex_absolute * norm_t

            loss = self.cross_entropy(complex_absolute, target)

            idx_l = target - self.delta
            idx_l[idx_l.le(0)] = 0
            idx_r = target + self.delta
            idx_r[idx_r.ge(self.high_bound - self.low_bound - 1)] = self.high_bound - self.low_bound - 1

            loss_snr = 0.0
            for i in range(0, batch_size):
                loss_snr = loss_snr + 1 - torch.sum(complex_absolute[i, idx_l[i]:idx_r[i]])

            loss_snr = loss_snr / batch_size

            loss = loss + loss_snr

        return loss, whole_max_idx
