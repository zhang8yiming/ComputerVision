import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):

    def __init__(self, in_chans, out_chans, rate=1):
        super(ASPP, self).__init__()
        # your code
        if rate == 1:
            dilations = [1, 6, 12, 18]
        if rate == 2:
            dilations = [1, 12, 24, 36]

        print('inchans: ', in_chans)
        print('out_chans', out_chans)
    
        self.aspp1 = nn.Sequential(nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=1, padding=0, dilation=dilations[0],bias=False),
                                   nn.BatchNorm2d(out_chans),
                                   nn.ReLU()
                                   )
        self.aspp2 = nn.Sequential(nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1],bias=False),
                                   nn.BatchNorm2d(out_chans),
                                   nn.ReLU()
                                   )
        self.aspp3 = nn.Sequential(nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2],bias=False),
                                   nn.BatchNorm2d(out_chans),
                                   nn.ReLU()
                                   )
        self.aspp4 = nn.Sequential(nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=dilations[3], dilation=dilations[3],bias=False),
                                   nn.BatchNorm2d(out_chans),
                                   nn.ReLU()
                                   )
        
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=1, bias=True),
                                            #  nn.BatchNorm2d(out_chans),
                                            #  nn.ReLU()
                                             )
        
        self.output = nn.Sequential(nn.Conv2d(out_chans*5, out_chans, kernel_size=1, padding=0, stride=1, bias=False),
                                    nn.BatchNorm2d(out_chans),
                                    nn.ReLU()
                                    )
        

    def forward(self, x):
        # your code
        x1 = self.aspp1(x)
        # print('x1 shape: ', x1.shape)
        x2 = self.aspp2(x)
        # print('x2 shape: ', x2.shape)
        x3 = self.aspp3(x)
        # print('x3 shape: ', x3.shape)
        x4 = self.aspp4(x)
        # print('x4 shape: ', x4.shape)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear')
        # print('x5 shape: ', x5.shape)
        result = self.output(torch.cat((x1, x2, x3, x4, x5), dim=1))

        return result

if __name__ == '__main__':
    model = ASPP(256, 256,1)

    rgb = torch.randn(1, 256,  100, 100)

    out = model(rgb)

    # print(out.size())
