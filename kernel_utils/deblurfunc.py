from abc import ABC, abstractmethod
from motionblur.motionblur import Kernel
from kernel_utils.img_utils import Blurkernel, DeblurKernel, perform_tilt

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)

class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def forward(self, data, **kwargs):
        # A^T * A 
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)

class GaussianBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

class GaussianDeblurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = DeblurKernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

class DeblurOperator(LinearOperator):
    def __init__(self, kernel_size, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = DeblurKernel(kernel_size=kernel_size,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

class BlindBlurOperator(LinearOperator):
    def __init__(self, device, **kwargs) -> None:
        self.device = device
    
    def forward(self, data, kernel, **kwargs):
        return self.apply_kernel(data, kernel)

    def transpose(self, data, **kwargs):
        return data
    
    def apply_kernel(self, data, kernel):
        #TODO: faster way to apply conv?:W
        
        b_img = torch.zeros_like(data).to(self.device)
        for i in range(3):
            b_img[:, i, :, :] = F.conv2d(data[:, i:i+1, :, :], kernel, padding='same')
        return b_img

class ResBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size, bias=True, norm=False):
        # conv=common_default_conv, norm=False, act=default_act):

        super(ResBlock, self).__init__()
        
        modules = []
        for i in range(2):
            modules.append(nn.Conv2d(n_feats, n_feats, 3, padding=(1,), bias=True, groups=1))
            if norm: modules.append(norm(self.n_feats))
            if self.default_act and i == 0: modules.append(self.default_act())

        self.body = nn.Sequential(*modules)
    
    def default_act(self):
        return nn.ReLU(True)
    
    def forward(self, x):
        res = self.body(x)
        res += x

        return res
    

class ToyDeblurFunc(nn.Module):
    def __init__(self, args, in_channels=3, out_channels=3, n_feats=None, kernel_size=3, n_resblocks=1, rgb_range=256, mean_shift=True):
        super(ToyDeblurFunc, self).__init__()

        self.image_size = args.image_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feats = args.n_feats
        self.kernel_size = args.kernel_size if kernel_size is None else kernel_size
        self.n_resblocks = args.n_resblocks if n_resblocks is None else n_resblocks

        self.mean_shift = mean_shift
        self.rgb_range = args.rgb_range if rgb_range is None else rgb_range
        self.mean = self.rgb_range / 2

        # # 7conv
        self.conv0 = nn.Conv2d(self.in_channels, self.n_feats, 3, padding=(1,), bias=True, groups=1)
        self.conv1 = nn.Conv2d(self.n_feats, self.n_feats, 3, padding=(1,), bias=True, groups=1)
        self.conv2 = nn.Conv2d(self.n_feats, self.n_feats, 3, padding=(1,), bias=True, groups=1)
        self.conv3 = nn.Conv2d(self.n_feats, self.n_feats, 3, padding=(1,), bias=True, groups=1)
        self.conv4 = nn.Conv2d(self.n_feats, self.n_feats, 3, padding=(1,), bias=True, groups=1)
        self.conv5 = nn.Conv2d(self.n_feats, self.n_feats, 3, padding=(1,), bias=True, groups=1)
        self.conv6 = nn.Conv2d(self.n_feats, self.n_feats, 3, padding=(1,), bias=True, groups=1)
        self.conv7 = nn.Conv2d(self.n_feats, self.in_channels, 3, padding=(1,), bias=True, groups=1)


        # # Residual
        # modules = []
        # modules.append(nn.Conv2d(self.in_channels, self.n_feats, 3, padding=(1,), bias=True, groups=1))
        # for _ in range(self.n_resblocks):
        #     modules.append(ResBlock(self.n_feats, 3))
        # modules.append(nn.Conv2d(self.n_feats, self.in_channels, 3, padding=(1,), bias=True, groups=1))

        # self.body = nn.Sequential(*modules)


    def forward(self, input): 
        x = self.conv0(input)
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        return x
   
    # def forward(self, input): # Residual
    #     if self.mean_shift:
    #         input = input - self.mean

    #     output = self.body(input)

    #     if self.mean_shift:
    #         output = output + self.mean

    #     return output