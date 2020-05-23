#!/usr/bin/env python
import glob

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import flowiz as fz

##########################################################

assert (int(
    str('').join(torch.__version__.split('.')[0:3]).split('+')[0]) >= 41)  # requires at least pytorch version 0.4.1

torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'sintel-final'
# arguments_strFirst = './images/first.png'
# arguments_strSecond = './images/second.png'
arguments_strOut = './out.flo'

for strOption, strArgument in \
        getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--model' and strArgument != '': arguments_strModel = strArgument  # which model to use, see below
    if strOption == '--first' and strArgument != '': arguments_strFirst = strArgument  # path to the first frame
    if strOption == '--second' and strArgument != '': arguments_strSecond = strArgument  # path to the second frame
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument  # path to where the output should
    # be stored
# end

##########################################################

Backward_tensorGrid = {}


def get_backward(tensor_input, tensor_flow):
    if str(tensor_flow.size()) not in Backward_tensorGrid:
        tensor_horizontal = torch.linspace(-1.0, 1.0, tensor_flow.size(3)).view(1, 1, 1, tensor_flow.size(3)).expand(
            tensor_flow.size(0), -1, tensor_flow.size(2), -1)
        tensor_vertical = torch.linspace(-1.0, 1.0, tensor_flow.size(2)).view(1, 1, tensor_flow.size(2), 1).expand(
            tensor_flow.size(0), -1, -1, tensor_flow.size(3))

        Backward_tensorGrid[str(tensor_flow.size())] = torch.cat([tensor_horizontal, tensor_vertical], 1).cuda()
    # end

    tensor_flow = torch.cat([tensor_flow[:, 0:1, :, :] / ((tensor_input.size(3) - 1.0) / 2.0),
                             tensor_flow[:, 1:2, :, :] / ((tensor_input.size(2) - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tensor_input,
                                           grid=(Backward_tensorGrid[str(tensor_flow.size())] + tensor_flow).permute(0,
                                                                                                                     2,
                                                                                                                     3,
                                                                                                                     1),
                                           mode='bilinear', padding_mode='border', align_corners=True)


# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        class Preprocess(torch.nn.Module):
            def __init__(self):
                super(Preprocess, self).__init__()

            # end

            def forward(self, n_tensor_input):
                tensor_blue = (n_tensor_input[:, 0:1, :, :] - 0.406) / 0.225
                tensor_green = (n_tensor_input[:, 1:2, :, :] - 0.456) / 0.224
                tensor_red = (n_tensor_input[:, 2:3, :, :] - 0.485) / 0.229

                return torch.cat([tensor_red, tensor_green, tensor_blue], 1)

        # end

        # end

        class Basic(torch.nn.Module):
            def __init__(self, int_level):
                super(Basic, self).__init__()

                self.moduleBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )

            # end

            def forward(self, d_tensor_input):
                return self.moduleBasic(d_tensor_input)

        # end

        # end

        self.modulePreprocess = Preprocess()

        self.moduleBasic = torch.nn.ModuleList([Basic(intLevel) for intLevel in range(6)])

        self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))

    # end

    def forward(self, f_tensor_first, f_tensor_second):

        f_tensor_first = [self.modulePreprocess(f_tensor_first)]
        f_tensor_second = [self.modulePreprocess(f_tensor_second)]

        for intLevel in range(5):
            if f_tensor_first[0].size(2) > 32 or f_tensor_first[0].size(3) > 32:
                f_tensor_first.insert(0,
                                      torch.nn.functional.avg_pool2d(input=f_tensor_first[0], kernel_size=2, stride=2,
                                                                     count_include_pad=False))
                f_tensor_second.insert(0,
                                       torch.nn.functional.avg_pool2d(input=f_tensor_second[0], kernel_size=2, stride=2,
                                                                      count_include_pad=False))
        # end
        # end

        tensor_flow = f_tensor_first[0].new_zeros(
            [f_tensor_first[0].size(0), 2, int(math.floor(f_tensor_first[0].size(2) / 2.0)),
             int(math.floor(f_tensor_first[0].size(3) / 2.0))])

        for intLevel in range(len(f_tensor_first)):
            tensor_upsampled = torch.nn.functional.interpolate(input=tensor_flow, scale_factor=2, mode='bilinear',
                                                               align_corners=True) * 2.0

            if tensor_upsampled.size(2) != f_tensor_first[intLevel].size(2): tensor_upsampled = torch.nn.functional.pad(
                input=tensor_upsampled, pad=[0, 0, 0, 1], mode='replicate')
            if tensor_upsampled.size(3) != f_tensor_first[intLevel].size(3): tensor_upsampled = torch.nn.functional.pad(
                input=tensor_upsampled, pad=[0, 1, 0, 0], mode='replicate')

            tensor_flow = self.moduleBasic[intLevel](torch.cat(
                [f_tensor_first[intLevel],
                 get_backward(tensor_input=f_tensor_second[intLevel], tensor_flow=tensor_upsampled),
                 tensor_upsampled], 1)) + tensor_upsampled
        # end

        return tensor_flow


# end


# end

moduleNetwork = Network().cuda().eval()


##########################################################

def estimate(tensor_first, tensor_second):
    assert (tensor_first.size(1) == tensor_second.size(1))
    assert (tensor_first.size(2) == tensor_second.size(2))

    int_width = tensor_first.size(2)
    int_height = tensor_first.size(1)

    tensor_preprocessed_first = tensor_first.cuda().view(1, 3, int_height, int_width)
    tensor_preprocessed_second = tensor_second.cuda().view(1, 3, int_height, int_width)

    int_preprocessed_width = int(math.floor(math.ceil(int_width / 32.0) * 32.0))
    int_preprocessed_height = int(math.floor(math.ceil(int_height / 32.0) * 32.0))

    tensor_preprocessed_first = torch.nn.functional.interpolate(input=tensor_preprocessed_first,
                                                                size=(int_preprocessed_height, int_preprocessed_width),
                                                                mode='bilinear', align_corners=False)
    tensor_preprocessed_second = torch.nn.functional.interpolate(input=tensor_preprocessed_second,
                                                                 size=(int_preprocessed_height, int_preprocessed_width),
                                                                 mode='bilinear', align_corners=False)

    tensor_flow = torch.nn.functional.interpolate(
        input=moduleNetwork(tensor_preprocessed_first, tensor_preprocessed_second),
        size=(int_height, int_width), mode='bilinear', align_corners=False)

    tensor_flow[:, 0, :, :] *= float(int_width) / float(int_preprocessed_width)
    tensor_flow[:, 1, :, :] *= float(int_height) / float(int_preprocessed_height)

    return tensor_flow[0, :, :, :].cpu()


# end

##########################################################
def get_optical_flow(first_img, second_image):
    first_tensor = torch.FloatTensor(
        numpy.array(first_img)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
    second_tensor = torch.FloatTensor(
        numpy.array(second_image)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
    tensor_output = estimate(first_tensor, second_tensor)

    object_output = open(arguments_strOut, 'wb')

    numpy.array([80, 73, 69, 72], numpy.uint8).tofile(object_output)
    numpy.array([tensor_output.size(2), tensor_output.size(1)], numpy.int32).tofile(object_output)
    numpy.array(tensor_output.detach().numpy().transpose(1, 2, 0), numpy.float32).tofile(object_output)

    object_output.close()

    files = glob.glob(arguments_strOut)
    img = fz.convert_from_file(files[0])

    # print('returning flo: ')

    return img
