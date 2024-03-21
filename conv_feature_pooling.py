import math

import torch
from torch import nn
import numpy as np
from torch.nn import Conv2d, Sequential, BatchNorm2d, LeakyReLU, Flatten, Dropout, Module

from PT import inv_sqrtm, geodesic, riemannian_mean, sqrtm
from SPD_DNN import StiefelParameter, SPDParameter
from sklearn.covariance import GraphicalLassoCV
from PT import log


device = "cuda:0"


def GetPearson(martix):
    output = torch.corrcoef(martix)

    if torch.isnan(output).any:
        output = torch.where(torch.isnan(output), 0, output)
        mask = torch.eye(n=116)
        output = mask + (1 - mask) * output

    output_martix = output

    disturb=torch.eye(n=116)*1e-4
    output_martix=output_martix+disturb



    return output_martix.to(device)


def Get_geodesic_distance(martix_a, martix_b):
    distance = (torch.norm((log(martix_a) - log(martix_b)), p="fro", dim=(-2, -1)))
    distance = distance ** 2
    distance = distance.mean(dim=0)
    return distance


class SPDTransform(nn.Module):

    def __init__(self, input_size, output_size, in_channels=1):
        super(SPDTransform, self).__init__()

        if in_channels > 1:
            self.weight = StiefelParameter(torch.FloatTensor(in_channels, input_size, output_size), requires_grad=True)
        else:
            self.weight = StiefelParameter(torch.FloatTensor(input_size, output_size), requires_grad=True)

        nn.init.orthogonal_(self.weight)

    def forward(self, input):
        weight = self.weight

        output = weight.transpose(-2, -1) @ input @ weight
        return output


class SPDRectified(nn.Module):

    def __init__(self, epsilon=1e-4):
        super(SPDRectified, self).__init__()
        self.register_buffer('epsilon', torch.FloatTensor([epsilon]))

    def forward(self, input):
        global s, u
        try:
            s, u = torch.linalg.eigh(input)
        except:
            print(input)
            torch.save(input, 'error.pt')
        s = s.clamp(min=self.epsilon[0])

        s = s.diag_embed()
        output = u @ s @ u.transpose(-2, -1)

        return output
