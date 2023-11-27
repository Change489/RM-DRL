import torch
from torch import nn
import torch.nn.functional as F

from PT import log
from conv_feature_pooling import Get_geodesic_distance
import numpy as np

device = "cuda:0"


def Uncertain_Loss(preds):
    preds = torch.softmax(preds, dim=1)
    preds = preds[:, 0]
    loss = 1 - torch.abs(2 * preds - 1) ** 2
    loss = torch.mean(loss, dim=0)

    return loss


def Conv_Feature_Distance_Loss(features, labels):
    batch_num = features.shape[0]
    loss = 0
    center = torch.mean(features, dim=0)
    for i in range(batch_num):
        cur_pred = features[i]
        cur_label = labels[i]
        mse_loss = nn.MSELoss()
        for j in range(batch_num):

            if i == j:
                continue
            else:
                
                if torch.equal(cur_label, labels[j]):
                    result = mse_loss((cur_pred), (features[j]))
                    result = torch.mean(result, dim=0)
                    loss += result
                
                if not torch.equal(cur_label, labels[j]):
                    result = -mse_loss((cur_pred), (features[j]))
                    result = torch.mean(result, dim=0)
                    loss += result

        return loss / batch_num


def Feature_Informative_Loss(features, labels):
    batch_num = features.shape[0]
    loss = 0
    pos_loss = 0
    neg_loss = 0
    pos_num = 0
    neg_num = 0
    anchor = 0

    cur_pred = features[anchor]
    cur_label = labels[anchor]

    for j in range(batch_num):

        if anchor == j:
            continue
        else:

            
            if torch.equal(cur_label, labels[j]):
                result = Get_geodesic_distance(cur_pred, (features[j]))

                pos_loss += result
                pos_num += 1
            
            if not torch.equal(cur_label, labels[j]):
                result = -Get_geodesic_distance(cur_pred, (features[j]))
                neg_loss += result
                neg_num += 1

    if pos_num != 0:
        pos_loss = pos_loss / pos_num
        loss += pos_loss
    if neg_num != 0:
        neg_loss = neg_loss / neg_num
        loss += neg_loss

    return loss


def Feature_Distance_Loss(features, labels):
    batch_num = features.shape[0]
    loss = 0
    pos_loss = 0
    neg_loss = 0
    pos_num = 0
    neg_num = 0
    flag = torch.eye(n=batch_num, dtype=torch.int)

    for i in range(batch_num):

        cur_pred = features[i]
        cur_label = labels[i]

        for j in range(batch_num):

            if flag[i][j] == 1:
                continue

            else:

                if i == j:
                    continue
                else:

                    
                    if torch.equal(cur_label, labels[j]):
                        result = Get_geodesic_distance(cur_pred, (features[j]))
                        pos_loss += result
                        pos_num += 1
                    
                    if not torch.equal(cur_label, labels[j]):
                        result = -Get_geodesic_distance(cur_pred, (features[j]))
                        neg_loss += result
                        neg_num += 1

                    flag[i][j] = 1
                    flag[j][i] = 1

    if pos_num != 0:
        pos_loss = pos_loss / pos_num
        
    if neg_num != 0:
        neg_loss = neg_loss / neg_num


    return pos_loss + neg_loss


def Feature_Diff_Loss(co, sp):
    batch_num = co.shape[0]

    co = log(co)
    sp = log(sp)

    co = torch.flatten(co, 1)
    sp = torch.flatten(sp, 1)

    co = F.normalize(co, p=2, dim=1)
    co = co.reshape(batch_num, 1, -1)

    sp = F.normalize(sp, p=2, dim=1)
    sp = sp.reshape(batch_num, -1, 1)

    loss = torch.bmm(co, sp)
    loss = torch.abs(loss)
    loss = loss.mean(dim=0)

    return loss

































