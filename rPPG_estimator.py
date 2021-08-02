import torch
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(
            self,
            inplanes,
            planes,
            win_len,
            roi_num,
    ):
        super(AttentionBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.win_len = win_len
        self.roi_num = roi_num
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

        self.part1_gap = nn.AvgPool2d(kernel_size=(self.win_len, 1))
        self.part1_fc1 = nn.Sequential(
            nn.Linear(self.roi_num, self.roi_num),
            nn.BatchNorm1d(self.planes),
            nn.ReLU(inplace=True),
        )
        self.part1_fc2 = nn.Sequential(
            nn.Linear(self.roi_num, self.roi_num),
            nn.Sigmoid(),
        )

        self.part2_conv1 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)

        x_att = self.part1_gap(x)
        x_att = x_att.view(x_att.shape[:2] + x_att.shape[3:])
        x_att = self.part1_fc1(x_att)
        x_att = self.part1_fc2(x_att)
        x_att = x_att.view(x_att.shape[:2] + (1,) + x_att.shape[2:])

        x = self.part2_conv1(x)

        x = x * x_att

        return x


class BaseBlock(nn.Module):
    def __init__(
            self,
            inplanes,
            planes,
            drop_rate=0.0,
    ):
        super(BaseBlock, self).__init__()
        self.drop_rate = drop_rate
        self.kernel_size = 3
        self.padding = 1

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(self.kernel_size, self.kernel_size), stride=(2, 2),
                               padding=(self.padding, self.padding), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if self.drop_rate > 0.0:
            self.drop1 = nn.Dropout2d(self.drop_rate)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(self.kernel_size, self.kernel_size), stride=(1, 1),
                               padding=(self.padding, self.padding), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if self.drop_rate > 0.0:
            self.drop2 = nn.Dropout2d(self.drop_rate)

        self.conv3 = nn.Conv2d(planes, planes, kernel_size=(self.kernel_size, self.kernel_size), stride=(1, 1),
                               padding=(self.padding, self.padding), bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        if self.drop_rate > 0.0:
            self.drop3 = nn.Dropout2d(self.drop_rate)

        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=(self.kernel_size, self.kernel_size), stride=(2, 2),
                         padding=(self.padding, self.padding)),
            nn.Conv2d(inplanes, planes, kernel_size=(1, 1,), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        if self.drop_rate > 0.0:
            out = self.drop1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.drop_rate > 0.0:
            out = self.drop2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.drop_rate > 0.0:
            out = self.drop3(out)

        residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class rPPG_2D_Estimator(nn.Module):
    def __init__(self, num_frames=300, drop_rate=0.0):
        super(rPPG_2D_Estimator, self).__init__()
        self.num_frames = num_frames
        self.drop_rate = drop_rate

        self.attention_block1 = AttentionBlock(inplanes=3, planes=64, win_len=300, roi_num=25)

        self.basic_block1 = BaseBlock(inplanes=64, planes=64, drop_rate=self.drop_rate)

        self.attention_block2 = AttentionBlock(inplanes=64, planes=64, win_len=150, roi_num=13)

        self.basic_block2 = BaseBlock(inplanes=64, planes=128, drop_rate=self.drop_rate)

        self.attention_block3 = AttentionBlock(inplanes=128, planes=128, win_len=75, roi_num=7)

        self.basic_block3 = BaseBlock(inplanes=128, planes=256, drop_rate=self.drop_rate)

        self.attention_block4 = AttentionBlock(inplanes=256, planes=256, win_len=38, roi_num=4)

        self.basic_block4 = BaseBlock(inplanes=256, planes=256, drop_rate=self.drop_rate)

        self.attention_block5 = AttentionBlock(inplanes=256, planes=256, win_len=19, roi_num=2)

        self.pool1 = nn.AdaptiveAvgPool2d((19, 1))
        #
        # estimate sig
        #
        self.up_block1 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=(4,), stride=(4,)),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
        )

        self.up_block2 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=(2,), stride=(2,)),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
        )

        self.up_block3 = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=(2,), stride=(2,)),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
        )

        self.up_final = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv1d(64, 1, kernel_size=(1,), stride=(1,), padding=(0,)),
            nn.Sigmoid(),
        )

        self._initialize()

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view([batch_size, -1, 300, 25])

        x = self.attention_block1(x)
        x = self.basic_block1(x)

        x = self.attention_block2(x)
        x = self.basic_block2(x)

        x = self.attention_block3(x)
        x = self.basic_block3(x)

        x = self.attention_block4(x)
        x = self.basic_block4(x)

        x = self.attention_block5(x)

        x = self.pool1(x)

        x = x.view([batch_size, -1, 19])

        x_sig = self.up_block1(x)

        x_sig = x_sig[:, :, :-1]

        x_sig = self.up_block2(x_sig)

        x_sig = self.up_block3(x_sig)

        x_sig = self.up_final(x_sig)

        x_sig = x_sig.view([batch_size, self.num_frames])

        return x_sig

    def _initialize(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                nn.init.constant_(m.bias, 0.0)
            elif classname.find('Conv') != -1:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif classname.find('BatchNorm') != -1:
                if m.affine:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)


if __name__ == '__main__':
    ipt = torch.rand([2, 3, 300, 5, 5])
    mdl = rPPG_2D_Estimator(num_frames=300, drop_rate=0.3)
    opt = mdl(ipt)
    print(opt.shape)
    print(sum([_.numel() for _ in mdl.parameters()]))
