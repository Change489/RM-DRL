from torch.nn import Linear
import os
import sys
from torch.nn import Linear
from conv_feature_pooling import *

BASE_DIR = os.path.dirname(sys.path[0])
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR))

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from SPD_DNN.optimizer import StiefelMetaOptimizer
from dataset import *
from loss import *

from PT import log

device = "cuda:0"


class RM_DRL(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = Flatten()

        self.layer_1 = nn.Sequential(

            SPDTransform(116, 96, 1),

            SPDRectified(),

        )

        self.layer_2 = nn.Sequential(
            SPDTransform(96, 66, 1),

            SPDRectified()

        )

        self.layer_3 = nn.Sequential(
            SPDTransform(66, 26, 1),

        )

        self.co_disentangle = nn.Sequential(
            SPDTransform(26, 26, 1),

            SPDRectified()
        )

        self.sp_disentangle = nn.Sequential(
            SPDTransform(26, 26, 1),

            SPDRectified(),
        )

        self.spd_recon = nn.Sequential(
            SPDTransform(52, 26, 1),

        )

        self.fc_layers = nn.Sequential(

            Dropout(p=0.2),
            Linear(676, 256),
            Dropout(p=0.2),
            LeakyReLU(negative_slope=1e-2),
            Linear(256, 2),

        )

        # 参数初始化
        for m in self.modules():

            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=1e-2)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        global sp_feature, co_feature
        features = []


        input = input.unsqueeze(1).to(torch.float)

        embed_stage_1 = self.layer_1(input)
        features.append(embed_stage_1)
        embed_stage_2 = self.layer_2(embed_stage_1)
        features.append(embed_stage_2)
        embed_stage_3 = self.layer_3(embed_stage_2)
        features.append(embed_stage_3)

        co_feature = self.co_disentangle(embed_stage_3)

        features.append(co_feature)

        if self.training:
            sp_feature = self.sp_disentangle(embed_stage_3)

            features.append(sp_feature)

            diag = torch.zeros(size=(
                embed_stage_3.shape[0], embed_stage_3.shape[1], embed_stage_3.shape[2] * 2, embed_stage_3.shape[2] * 2))
            diag = diag.to(device)
            diag[:, :, 0:co_feature.shape[2], 0:co_feature.shape[2]] = co_feature
            diag[:, :, co_feature.shape[2]:, co_feature.shape[2]:] = sp_feature

            recon_spd = self.spd_recon(diag)

            features.append(recon_spd)


        output = log(co_feature)
        output = self.flatten(output)
        output = self.fc_layers(output)

        features.append(input)

        return output, features


if __name__ == '__main__':
    import torch
    import numpy as np
    import warnings
    import random
    import os

    warnings.filterwarnings("ignore")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    seed = 42
    random.seed(seed)
    np.random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

    
    test_data = torch.randn(size=(116, 80))
    test_data=GetPearson(test_data)
    
    test_data=test_data.unsqueeze(0)
    test_data=test_data.to(device)
    model = RM_DRL().to(device)
    output, feature = model(test_data)
    print(output)