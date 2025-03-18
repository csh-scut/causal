import sys
sys.path.append("..")
from data_process import data_loader, lorenz96
from model.config import Myconfig
from model.train2 import DataGenerator
import torch
import pandas as pd
import hdf5storage
import h5py
import os
from geopy.distance import geodesic
from model.unit import Mymodel1
import matplotlib.pyplot as plt
#解决中文显示问题
import numpy as np


# test sample size:90*50
# hid1:128*50
# hid2:64*50
# hid3:32*50
# out:16*50

# def predict_one_sample(model, batch_size, input):
#     model.eval()
#     # input.shape:90*50
#     x = torch.stack([input] * batch_size)
#     pred = model(x)
#     return pred

def calculate_lrp(model_named_parameters, x, hid_ls, lat_val, type):
    # x:90*50
    # hid_ls:len=3
    # lat_val:16*50
    for name, value in model_named_parameters:
        if name == "encoder_layer_x_"+type+".layer1.linear.weight":
            w1 = value #128*90
        elif name == "encoder_layer_x_"+type+".layer2.linear.weight":
            w2 = value # 64*128
        elif name == "encoder_layer_x_"+type+".layer3.linear.weight":
            w3 = value # 32*64
        elif name == "encoder_layer_x_"+type+".layer4.linear.weight":
            w4 = value # 16*32
            break
        else:
            continue
    hid_1 = hid_ls[0] #batch
    hid_2 = hid_ls[1]
    hid_3 = hid_ls[2]
    sample_score_ls = []
    for sample in range(50):
        sample_x = x[:, sample]  # 90
        hid_val_ls = [x[:, sample], hid_1[0][:, sample], hid_2[0][:, sample], hid_3[0][:, sample]]
        sample_u = lat_val[:, sample]  # 提取第i个样本的隐变量向量
        # sample_hid1 = hid_u1[:,sample] #128
        # sample_hid2 = hid_u2[:,sample] # 64
        # sample_hid3 = hid_u3[:,sample] # 32
        lv_ls = []
        for i in range(16):  # latent variable dimension
            val = sample_u[i]  # latent variable value
            # w * a
            sigma_w_a_ls = [w1 @ hid_val_ls[0],
                            w2 @ hid_val_ls[1],
                            w3 @ hid_val_ls[2],
                            (w4 @ hid_val_ls[3])[i]]
            w_a_ls = [w1 * (sample_x.expand_as(w1)),
                      w2 * (hid_val_ls[1].expand_as(w2)),
                      w3 * (hid_val_ls[2].expand_as(w3)),
                      w4[i, :] * (hid_val_ls[3])]
            # sigma_w_a_1 = w1 @ sample_x # 128
            # sigma_w_a_2 = w2 @ sample_hid1 #64
            # sigma_w_a_3 = w3 @ sample_hid2 #32
            # sigma_w_a_4 = w4 @ sample_hid3  #16
            rlp_score = val
            for j in range(3, -1, -1):  # num of hidden layer
                # 从后往前计算relevance score
                # message_between = []
                # sigma_j = sigma_w_a_ls[j]
                ratio_matrix = w_a_ls[j] / (sigma_w_a_ls[j].unsqueeze(dim=-1).expand_as(w_a_ls[j]))
                if j == 3:
                    rlp_score = rlp_score * ratio_matrix
                else:
                    rlp_score = (rlp_score.unsqueeze(dim=-1).expand_as(ratio_matrix) * ratio_matrix).sum(dim=0)
            lv_ls.append(rlp_score / val)
        avg_score = torch.stack(lv_ls).mean(dim=0)
        sample_score_ls.append(avg_score)
    result = torch.stack(sample_score_ls).mean(dim=0)
    return result

def calculate_lrp_3layers(model_named_parameters, x, hid_ls, lat_val, type, config):
    # x:90*50
    # hid_ls:len=3
    # lat_val:16*50
    for name, value in model_named_parameters:
        if name == "encoder_layer_x_" + type + ".layer1.linear.weight":
            w1 = value  # 128*90
        elif name == "encoder_layer_x_" + type + ".layer1.linear.bias":
            b1 = value  # 128
        elif name == "encoder_layer_x_" + type + ".layer2.linear.weight":
            w2 = value  # 64*128
        elif name == "encoder_layer_x_" + type + ".layer2.linear.bias":
            b2 = value  # 64
        elif name == "encoder_layer_x_" + type + ".layer3.linear.weight":
            w3 = value  # 32*64
        elif name == "encoder_layer_x_" + type + ".layer3.linear.bias":
            b3 = value  # 32
            break
        else:
            continue
    hid_1 = hid_ls[0] #batch
    hid_2 = hid_ls[1]
    # hid_3 = hid_ls[2]
    sample_score_ls = []
    sample_num = config.TRAIN_LEN
    lat_dim = config.HIDDEN_V_DIM
    epsilon = torch.tensor(0.0001)
    for sample in range(sample_num):
        sample_x = x[:, sample]  # 90
        hid_val_ls = [x[:, sample], hid_1[0][:, sample], hid_2[0][:, sample]]
        sample_u = lat_val[:, sample]  # 提取第i个样本的隐变量向量
        # sample_hid1 = hid_u1[:,sample] #128
        # sample_hid2 = hid_u2[:,sample] # 64
        # sample_hid3 = hid_u3[:,sample] # 32
        lv_ls = []
        for i in range(lat_dim):  # latent variable dimension
            val = sample_u[i]  # latent variable value
            # w * a
            sigma_w_a_ls = [w1 @ hid_val_ls[0],
                            w2 @ hid_val_ls[1],
                            (w3 @ hid_val_ls[2])[i]]
            w_a_ls = [w1 * (sample_x.expand_as(w1)),
                      w2 * (hid_val_ls[1].expand_as(w2)),
                      w3[i, :] * (hid_val_ls[2])]
            w_a_ls[0] = w_a_ls[0] \
                            + (epsilon.expand_as(sigma_w_a_ls[0]) *
                               torch.sign(sigma_w_a_ls[0])).unsqueeze(dim=-1).expand_as(w_a_ls[0]) / w1.shape[1]
                            # + b1.unsqueeze(dim=-1).expand_as(pos_w_a_ls[0]) / w1.shape[1]

            w_a_ls[1] = w_a_ls[1] \
                            + (epsilon.expand_as(sigma_w_a_ls[1]) *
                               torch.sign(sigma_w_a_ls[1])).unsqueeze(dim=-1).expand_as(w_a_ls[1]) / w2.shape[1]
            w_a_ls[2] = w_a_ls[2] \
                            + (epsilon.expand_as(sigma_w_a_ls[2]) *
                               torch.sign(sigma_w_a_ls[2])).unsqueeze(dim=-1).expand_as(w_a_ls[2]) / w3.shape[1]
            # sigma_w_a_1 = w1 @ sample_x # 128
            # sigma_w_a_2 = w2 @ sample_hid1 #64
            # sigma_w_a_3 = w3 @ sample_hid2 #32
            # sigma_w_a_4 = w4 @ sample_hid3  #16
            rlp_score = val
            for j in range(2, -1, -1):  # num of hidden layer
                # 从后往前计算relevance score
                # message_between = []
                # sigma_j = sigma_w_a_ls[j]
                ratio_matrix = w_a_ls[j] / (sigma_w_a_ls[j].unsqueeze(dim=-1).expand_as(w_a_ls[j]))
                if j == 2:
                    rlp_score = rlp_score * ratio_matrix
                else:
                    rlp_score = (rlp_score.unsqueeze(dim=-1).expand_as(ratio_matrix) * ratio_matrix).sum(dim=0)
            lv_ls.append(rlp_score / val)
        avg_score = torch.stack(lv_ls).mean(dim=0)
        sample_score_ls.append(avg_score)
    result = torch.stack(sample_score_ls).mean(dim=0)
    return result

def calculate_lrp_3layers_alpha1beta0(model_named_parameters, x, hid_ls, lat_val, type, config):
    # x:90*50
    # hid_ls:len=3
    # lat_val:16*50
    for name, value in model_named_parameters:
        if name == "encoder_layer_x_"+type+".layer1.linear.weight":
            w1 = value #128*90
        elif name == "encoder_layer_x_" + type + ".layer1.linear.bias":
            b1 = value  # 128
        elif name == "encoder_layer_x_"+type+".layer2.linear.weight":
            w2 = value # 64*128
        elif name == "encoder_layer_x_"+type+".layer2.linear.bias":
            b2 = value # 64
        elif name == "encoder_layer_x_"+type+".layer3.linear.weight":
            w3 = value # 32*64
        elif name == "encoder_layer_x_" + type + ".layer3.linear.bias":
            b3 = value  # 32
            break
        else:
            continue
    hid_1 = hid_ls[0] #batch
    hid_2 = hid_ls[1]
    # hid_3 = hid_ls[2]
    sample_score_ls = []
    sample_num = config.TRAIN_LEN
    lat_dim = config.HIDDEN_V_DIM
    b1[b1<0] = 0
    b2[b2<0] = 0
    b3[b3<0] = 0
    epsilon = 0.0001
    for sample in range(sample_num):
        sample_x = x[:, sample]  # 90
        hid_val_ls = [x[:, sample], hid_1[0][:, sample], hid_2[0][:, sample]]
        sample_u = lat_val[:, sample]  # 提取第i个样本的隐变量向量
        # sample_hid1 = hid_u1[:,sample] #128
        # sample_hid2 = hid_u2[:,sample] # 64
        # sample_hid3 = hid_u3[:,sample] # 32
        lv_ls = []
        for i in range(lat_dim):  # latent variable dimension
            val = sample_u[i]  # latent variable value
            # w * a
            # sigma_w_a_ls = [w1 @ hid_val_ls[0],
            #                 w2 @ hid_val_ls[1],
            #                 (w3 @ hid_val_ls[2])[i]]
            w_a_ls = [w1 * (sample_x.expand_as(w1)),
                      w2 * (hid_val_ls[1].expand_as(w2)),
                      w3[i, :] * (hid_val_ls[2])]
            # last_linear_wij = w3[i,:] #64
            # last_linear_wij[last_linear_wij<0] = 0
            # last_linear_wij = last_linear_wij/last_linear_wij.sum()
            # if sample == 0 and i == 0:
            #     print(last_linear_wij)
            pos_w_a_ls = w_a_ls.copy()
            pos_w_a_ls[0][pos_w_a_ls[0] < 0] = 0
            pos_w_a_ls[1][pos_w_a_ls[1] < 0] = 0
            pos_w_a_ls[2][pos_w_a_ls[2] < 0] = 0

            pos_w_a_ls[0] = pos_w_a_ls[0] \
                            + epsilon / w1.shape[1]\
                            + b1.unsqueeze(dim=-1).expand_as(pos_w_a_ls[0]) / w1.shape[1]

            pos_w_a_ls[1] = pos_w_a_ls[1] \
                            + epsilon / w2.shape[1]\
                            + b2.unsqueeze(dim=-1).expand_as(pos_w_a_ls[1]) / w2.shape[1]

            pos_w_a_ls[2] = pos_w_a_ls[2] \
                            + epsilon / w3.shape[1]\
                            + b3[i].expand_as(pos_w_a_ls[2]) / w3.shape[1]



            sigma_pos_w_a_ls = [pos_w_a_ls[0].sum(dim=1),
                                pos_w_a_ls[1].sum(dim=1),
                                pos_w_a_ls[2].sum()]
            # sigma_w_a_1 = w1 @ sample_x # 128
            # sigma_w_a_2 = w2 @ sample_hid1 #64
            # sigma_w_a_3 = w3 @ sample_hid2 #32
            # sigma_w_a_4 = w4 @ sample_hid3  #16
            rlp_score = val
            for j in range(2, -1, -1):  # num of hidden layer
                # 从后往前计算relevance score
                # message_between = []
                # sigma_j = sigma_w_a_ls[j]
                ratio_matrix = pos_w_a_ls[j] / (sigma_pos_w_a_ls[j].unsqueeze(dim=-1).expand_as(pos_w_a_ls[j]))
                if j == 2:
                    rlp_score = rlp_score * ratio_matrix
                    # rlp_score = last_linear_wij * rlp_score
                else:
                    rlp_score = (rlp_score.unsqueeze(dim=-1).expand_as(ratio_matrix) * ratio_matrix).sum(dim=0)
            lv_ls.append(rlp_score / val)
        avg_score = torch.stack(lv_ls).mean(dim=0)
        sample_score_ls.append(avg_score)
    result = torch.stack(sample_score_ls).mean(dim=0)
    return result#,lv_ls

def return_pos_neg(b):
    pos_b = b.clone()
    neg_b = b.clone()
    pos_b[b<0] = 0
    neg_b[b>0] = 0
    return pos_b,neg_b

def add_regular(w_a, b, type,epsilon,last=False):
    if type == "neg":
        flag = -1
    else:
        flag = 1
    if not last:
        nonzero_num = torch.count_nonzero(w_a, dim=1) # e.g. 4*128
        adjusted_nonzero_num = torch.where(nonzero_num==0,1,nonzero_num)
        # b = b.expand_as(non_zero_num) # e.g. 4*128
        # print((b / adjusted_nonzero_num).shape)
        w_a = w_a + (flag * torch.sign(w_a)) * \
              (b / adjusted_nonzero_num + flag * (epsilon.expand_as(b)) / adjusted_nonzero_num).unsqueeze(dim=-1).expand_as(w_a) # e.g. 4*128*90
        # for i in range(len(nonzero_num)):
        #     if nonzero_num[i] == 0:
        #         continue
        #     else:
        #         w_a[i,:] = w_a[i,:] + (flag * torch.sign(w_a[i,:])) * \
        #                    (b[i]/nonzero_num[i] + flag*epsilon/nonzero_num[i]).expand_as(w_a[i,:])
        return w_a
    else:
        nonzero_num = torch.count_nonzero(w_a) #w_a:64 
        if nonzero_num != 0:
            w_a = w_a + (flag * torch.sign(w_a)) * \
                  (b/nonzero_num + flag*epsilon/nonzero_num).expand_as(w_a)
        return w_a

def calculate_lrp_3layers_alphabeta(model_named_parameters, x, hid_ls, lat_val, type, config, device, alpha=1,beta=0):
    # x:90*50
    # hid_ls:len=3
    # lat_val:16*50
    # x.to(device)
    # print(x.device)
    for name, value in model_named_parameters:
        if name == "encoder_layer_x_"+type+".layer1.linear.weight":
            w1 = value #128*90
        elif name == "encoder_layer_x_" + type + ".layer1.linear.bias":
            b1 = value  # 128
        elif name == "encoder_layer_x_"+type+".layer2.linear.weight":
            w2 = value # 64*128
        elif name == "encoder_layer_x_"+type+".layer2.linear.bias":
            b2 = value # 64
        elif name == "encoder_layer_x_"+type+".layer3.linear.weight":
            w3 = value # 32*64
        elif name == "encoder_layer_x_" + type + ".layer3.linear.bias":
            b3 = value  # 32
            break
        else:
            continue
    hid_1 = hid_ls[0] #batch
    hid_2 = hid_ls[1]
    # hid_3 = hid_ls[2]
    sample_score_ls = []
    sample_num = config.TRAIN_LEN
    lat_dim = config.HIDDEN_V_DIM
    pos_b1,neg_b1 = return_pos_neg(b1)
    pos_b2, neg_b2 = return_pos_neg(b2)
    pos_b3, neg_b3 = return_pos_neg(b3)
    epsilon = torch.tensor(0.0001).to(device)
    for sample in range(sample_num):
        sample_x = x[:, sample]  # 90
        hid_val_ls = [x[:, sample], hid_1[0][:, sample], hid_2[0][:, sample]]
        sample_u = lat_val[:, sample]  # 提取第i个样本的隐变量向量
        # sample_hid1 = hid_u1[:,sample] #128
        # sample_hid2 = hid_u2[:,sample] # 64
        # sample_hid3 = hid_u3[:,sample] # 32
        lv_ls = []
        for i in range(lat_dim):  # latent variable dimension
            val = sample_u[i]  # latent variable value

            w_a_ls = [w1 * (sample_x.expand_as(w1)),
                      w2 * (hid_val_ls[1].expand_as(w2)),
                      w3[i, :] * (hid_val_ls[2])]

            pos_w_a_ls = [return_pos_neg(w_a)[0] for w_a in w_a_ls]
            neg_w_a_ls = [return_pos_neg(w_a)[1] for w_a in w_a_ls]
            # print(pos_w_a_ls,neg_w_a_ls,sep="/n")
            # pos_w_a_ls[0][pos_w_a_ls[0] < 0] = 0
            # pos_w_a_ls[1][pos_w_a_ls[1] < 0] = 0
            # pos_w_a_ls[2][pos_w_a_ls[2] < 0] = 0
            # print((pos_b1/torch.count_nonzero(pos_w_a_ls[0],dim=1)).unsqueeze(dim=-1).expand_as(pos_w_a_ls[0]).shape)
            pos_w_a_ls[0] = add_regular(pos_w_a_ls[0],pos_b1,"pos",epsilon)
            # print(pos_w_a_ls[0])
            pos_w_a_ls[1] = add_regular(pos_w_a_ls[1],pos_b2,"pos",epsilon)
            pos_w_a_ls[2] = add_regular(pos_w_a_ls[2],pos_b3[i],"pos",epsilon,True)
            # print(pos_w_a_ls[2].shape)
            neg_w_a_ls[0] = add_regular(neg_w_a_ls[0],neg_b1,"neg",epsilon)
            neg_w_a_ls[1] = add_regular(neg_w_a_ls[1],neg_b2,"neg",epsilon)
            neg_w_a_ls[2] = add_regular(neg_w_a_ls[2],neg_b3[i],"neg",epsilon,True)



            sigma_pos_w_a_ls = [pos_w_a_ls[0].sum(dim=1),
                                pos_w_a_ls[1].sum(dim=1),
                                pos_w_a_ls[2].sum()]
            sigma_pos_w_a_ls = [torch.where(e==0,1,e) for e in sigma_pos_w_a_ls]
            sigma_neg_w_a_ls = [neg_w_a_ls[0].sum(dim=1),
                                neg_w_a_ls[1].sum(dim=1),
                                neg_w_a_ls[2].sum()]
            sigma_neg_w_a_ls = [torch.where(e==0,1,e) for e in sigma_neg_w_a_ls]
            # sigma_w_a_1 = w1 @ sample_x # 128
            # sigma_w_a_2 = w2 @ sample_hid1 #64
            # sigma_w_a_3 = w3 @ sample_hid2 #32
            # sigma_w_a_4 = w4 @ sample_hid3  #16
            rlp_score = val
            for j in range(2, -1, -1):  # num of hidden layer
                # 从后往前计算relevance score
                # message_between = []
                # sigma_j = sigma_w_a_ls[j]
                pos_ratio_matrix = pos_w_a_ls[j] / (sigma_pos_w_a_ls[j].unsqueeze(dim=-1).expand_as(pos_w_a_ls[j]))
                neg_ratio_matrix = neg_w_a_ls[j] / (sigma_neg_w_a_ls[j].unsqueeze(dim=-1).expand_as(neg_w_a_ls[j]))
                # print(pos_ratio_matrix.shape,neg_ratio_matrix.shape)
                if j == 2:
                    rlp_score = rlp_score * (alpha * pos_ratio_matrix + beta * neg_ratio_matrix)
                    # rlp_score = last_linear_wij * rlp_score
                else:
                    rlp_score = (rlp_score.unsqueeze(dim=-1).expand_as(pos_ratio_matrix) *
                                 (alpha * pos_ratio_matrix + beta * neg_ratio_matrix)).sum(dim=0)
            lv_ls.append(rlp_score / val)
        avg_score = torch.stack(lv_ls).mean(dim=0)
        sample_score_ls.append(avg_score)
    result = torch.stack(sample_score_ls).mean(dim=0)
    return result#,lv_ls

def add_regular_faster(w_a, b, type,epsilon):
    if type == "neg":
        flag = -1
    else:
        flag = 1
    # if not last:
    nonzero_num = torch.count_nonzero(w_a, dim=-1) # e.g. 4*128
    adjusted_nonzero_num = torch.where(nonzero_num==0,1,nonzero_num)
    b = b.expand_as(nonzero_num) # e.g. 4*128
    # print((b / adjusted_nonzero_num).shape)
    w_a = w_a + (flag * torch.sign(w_a)) * \
            (b / adjusted_nonzero_num + flag * (epsilon.expand_as(b)) / adjusted_nonzero_num).unsqueeze(dim=-1).expand_as(w_a) # e.g. 4*128*90
    return w_a
    # else:
    #     nonzero_num = torch.count_nonzero(w_a,dim=1) #w_a:4*64 
    #     if nonzero_num != 0:
    #         w_a = w_a + (flag * torch.sign(w_a)) * \
    #               (b/nonzero_num + flag*epsilon/nonzero_num).expand_as(w_a)
    #     return w_a

def calculate_all_latdim_for_each_time(lat_dim, time_x, S_time_lat, Z_time_lat, S_hid_ls, Z_hid_ls, S_w_a, Z_w_a, S_b, Z_b, tim, alpha, beta):
    
    S_pos_b1,S_neg_b1 = return_pos_neg(S_b[0])
    S_pos_b2, S_neg_b2 = return_pos_neg(S_b[1])
    S_pos_b3, S_neg_b3 = return_pos_neg(S_b[2])
    Z_pos_b1,Z_neg_b1 = return_pos_neg(Z_b[0])
    Z_pos_b2, Z_neg_b2 = return_pos_neg(Z_b[1])
    Z_pos_b3, Z_neg_b3 = return_pos_neg(Z_b[2])
    w1_s,w2_s,w3_s = S_w_a
    w1_z,w2_z,w3_z = Z_w_a

    epsilon = torch.tensor(0.0001).to(device)

    S_hid_1, S_hid_2 = S_hid_ls #4*128*50, 4*64*50
    Z_hid_1, Z_hid_2 = Z_hid_ls #4*128*50, 4*64*50

    S_lv_score_ls = []
    Z_lv_score_ls = []

    for i in range(lat_dim):  # latent variable dimension
        S_val = S_time_lat[:, i]  # latent variable value: length=4
        Z_val = Z_time_lat[:, i]

        S_w_a_ls = [w1_s * (time_x.unsqueeze(dim=1).expand(-1,w1_s.shape[0],-1)),  # 4*128*90
                    w2_s * (S_hid_1[:, :, tim].unsqueeze(dim=1).expand(-1,w2_s.shape[0],-1)), # 4*64*128
                    w3_s[i, :] * (S_hid_2[:, :, tim])] # 4*(1*)64

        Z_w_a_ls = [w1_z * (time_x.unsqueeze(dim=1).expand(-1,w1_z.shape[0],-1)),  # 4*128*90
                    w2_z * (Z_hid_1[:, :, tim].unsqueeze(dim=1).expand(-1,w2_z.shape[0],-1)), # 4*64*128
                    w3_z[i, :] * (Z_hid_2[:, :, tim])] # 4*(1*)64

        S_pos_w_a_ls = [return_pos_neg(w_a)[0] for w_a in S_w_a_ls]
        S_neg_w_a_ls = [return_pos_neg(w_a)[1] for w_a in S_w_a_ls]
        Z_pos_w_a_ls = [return_pos_neg(w_a)[0] for w_a in Z_w_a_ls]
        Z_neg_w_a_ls = [return_pos_neg(w_a)[1] for w_a in Z_w_a_ls]
        # print(pos_w_a_ls,neg_w_a_ls,sep="/n")

        # print((pos_b1/torch.count_nonzero(pos_w_a_ls[0],dim=1)).unsqueeze(dim=-1).expand_as(pos_w_a_ls[0]).shape)
        S_pos_w_a_ls[0] = add_regular_faster(S_pos_w_a_ls[0],S_pos_b1,"pos",epsilon)
        S_pos_w_a_ls[1] = add_regular_faster(S_pos_w_a_ls[1],S_pos_b2,"pos",epsilon)
        S_pos_w_a_ls[2] = add_regular_faster(S_pos_w_a_ls[2],S_pos_b3[i],"pos",epsilon)
        S_neg_w_a_ls[0] = add_regular_faster(S_neg_w_a_ls[0],S_neg_b1,"neg",epsilon)
        S_neg_w_a_ls[1] = add_regular_faster(S_neg_w_a_ls[1],S_neg_b2,"neg",epsilon)
        S_neg_w_a_ls[2] = add_regular_faster(S_neg_w_a_ls[2],S_neg_b3[i],"neg",epsilon)

        Z_pos_w_a_ls[0] = add_regular_faster(Z_pos_w_a_ls[0],Z_pos_b1,"pos",epsilon)
        Z_pos_w_a_ls[1] = add_regular_faster(Z_pos_w_a_ls[1],Z_pos_b2,"pos",epsilon)
        Z_pos_w_a_ls[2] = add_regular_faster(Z_pos_w_a_ls[2],Z_pos_b3[i],"pos",epsilon)
        Z_neg_w_a_ls[0] = add_regular_faster(Z_neg_w_a_ls[0],Z_neg_b1,"neg",epsilon)
        Z_neg_w_a_ls[1] = add_regular_faster(Z_neg_w_a_ls[1],Z_neg_b2,"neg",epsilon)
        Z_neg_w_a_ls[2] = add_regular_faster(Z_neg_w_a_ls[2],Z_neg_b3[i],"neg",epsilon)



        S_sigma_pos_w_a_ls = [S_pos_w_a_ls[0].sum(dim=-1),
                            S_pos_w_a_ls[1].sum(dim=-1),
                            S_pos_w_a_ls[2].sum(dim=-1)]
        S_sigma_pos_w_a_ls = [torch.where(e==0,1,e) for e in S_sigma_pos_w_a_ls]
        S_sigma_neg_w_a_ls = [S_neg_w_a_ls[0].sum(dim=-1),
                            S_neg_w_a_ls[1].sum(dim=-1),
                            S_neg_w_a_ls[2].sum(dim=-1)]
        S_sigma_neg_w_a_ls = [torch.where(e==0,1,e) for e in S_sigma_neg_w_a_ls]

        Z_sigma_pos_w_a_ls = [Z_pos_w_a_ls[0].sum(dim=-1),
                            Z_pos_w_a_ls[1].sum(dim=-1),
                            Z_pos_w_a_ls[2].sum(dim=-1)]
        Z_sigma_pos_w_a_ls = [torch.where(e==0,1,e) for e in Z_sigma_pos_w_a_ls]
        Z_sigma_neg_w_a_ls = [Z_neg_w_a_ls[0].sum(dim=-1),
                            Z_neg_w_a_ls[1].sum(dim=-1),
                            Z_neg_w_a_ls[2].sum(dim=-1)]
        Z_sigma_neg_w_a_ls = [torch.where(e==0,1,e) for e in Z_sigma_neg_w_a_ls]

        S_rlp_score = S_val
        Z_rlp_score = Z_val
        for j in range(2, -1, -1):  # num of hidden layer
            # 从后往前计算relevance score
            # message_between = []
            # sigma_j = sigma_w_a_ls[j]
            S_pos_ratio_matrix = S_pos_w_a_ls[j] / (S_sigma_pos_w_a_ls[j].unsqueeze(dim=-1).expand_as(S_pos_w_a_ls[j]))
            S_neg_ratio_matrix = S_neg_w_a_ls[j] / (S_sigma_neg_w_a_ls[j].unsqueeze(dim=-1).expand_as(S_neg_w_a_ls[j]))
            Z_pos_ratio_matrix = Z_pos_w_a_ls[j] / (Z_sigma_pos_w_a_ls[j].unsqueeze(dim=-1).expand_as(Z_pos_w_a_ls[j]))
            Z_neg_ratio_matrix = Z_neg_w_a_ls[j] / (Z_sigma_neg_w_a_ls[j].unsqueeze(dim=-1).expand_as(Z_neg_w_a_ls[j]))
            # print(pos_ratio_matrix.shape,neg_ratio_matrix.shape)
            if j == 2:
                S_rlp_score = (S_rlp_score.unsqueeze(dim=-1).expand_as(S_pos_ratio_matrix) * 
                            (alpha * S_pos_ratio_matrix + beta * S_neg_ratio_matrix))
                Z_rlp_score = (Z_rlp_score.unsqueeze(dim=-1).expand_as(Z_pos_ratio_matrix) * 
                            (alpha * Z_pos_ratio_matrix + beta * Z_neg_ratio_matrix))
                # rlp_score = last_linear_wij * rlp_score
            else:
                S_rlp_score = (S_rlp_score.unsqueeze(dim=-1).expand_as(S_pos_ratio_matrix) *
                            (alpha * S_pos_ratio_matrix + beta * S_neg_ratio_matrix)).sum(dim=1)
                Z_rlp_score = (Z_rlp_score.unsqueeze(dim=-1).expand_as(Z_pos_ratio_matrix) *
                            (alpha * Z_pos_ratio_matrix + beta * Z_neg_ratio_matrix)).sum(dim=1)
        S_lv_score_ls.append(S_rlp_score / S_val.unsqueeze(dim=-1).expand_as(S_rlp_score))
        Z_lv_score_ls.append(Z_rlp_score / Z_val.unsqueeze(dim=-1).expand_as(Z_rlp_score))

    S_avg_score = torch.stack(S_lv_score_ls).mean(dim=0) # 4*90
    Z_avg_score = torch.stack(Z_lv_score_ls).mean(dim=0) # 4*90

    return S_avg_score, Z_avg_score



def calculate_lrp_3layers_alphabeta_faster(model_named_parameters, x, S_hid_ls, Z_hid_ls, S_lat_val, Z_lat_val, config, device, alpha=2,beta=-1):
    # x:4*90*50
    # hid_ls:len=2
    # lat_val:4*32*50
    # x.to(device)
    # print(x.device)
    for name, value in model_named_parameters:
        if name == "encoder_layer_x_s.layer1.linear.weight":
            w1_s = value #128*90
        elif name == "encoder_layer_x_s.layer1.linear.bias":
            b1_s = value  # 128
        elif name == "encoder_layer_x_s.layer2.linear.weight":
            w2_s = value # 64*128
        elif name == "encoder_layer_x_s.layer2.linear.bias":
            b2_s = value # 64
        elif name == "encoder_layer_x_s.layer3.linear.weight":
            w3_s = value # 32*64
        elif name == "encoder_layer_x_s.layer3.linear.bias":
            b3_s = value  # 32
        elif name == "encoder_layer_x_z.layer1.linear.weight":
            w1_z = value #128*90
        elif name == "encoder_layer_x_z.layer1.linear.bias":
            b1_z = value  # 128
        elif name == "encoder_layer_x_z.layer2.linear.weight":
            w2_z = value # 64*128
        elif name == "encoder_layer_x_z.layer2.linear.bias":
            b2_z = value # 64
        elif name == "encoder_layer_x_z.layer3.linear.weight":
            w3_z = value # 32*64
        elif name == "encoder_layer_x_z.layer3.linear.bias":
            b3_z = value  # 32
        else:
            continue


    S_time_score_ls = []
    Z_time_score_ls = []

    time_num = config.TRAIN_LEN
    lat_dim = config.HIDDEN_V_DIM

    
    S_all_time_score_ls = []
    Z_all_time_score_ls = []
    for tim in range(time_num):
        time_x = x[:, :, tim]  # 4*90

        # hid_val_ls = [x[:, :, tim], hid_1[:, :, tim], hid_2[:, :, tim]]
        S_time_lat = S_lat_val[:, :, tim]  # 4*32
        Z_time_lat = Z_lat_val[:, :, tim]  # 4*32

        S_avg_score, Z_avg_score = calculate_all_latdim_for_each_time(lat_dim, time_x, S_time_lat, Z_time_lat, S_hid_ls, Z_hid_ls, 
                                                                      [w1_s,w2_s,w3_s], [w1_z,w2_z,w3_z], [b1_s,b2_s,b3_s], [b1_z,b2_z,b3_z],tim, alpha, beta)
        # S_lv_score_ls = []
        # Z_lv_score_ls = []

        # for i in range(lat_dim):  # latent variable dimension
        #     S_val = S_time[:, i]  # latent variable value: length=4
        #     Z_val = Z_time[:, i]
 
        #     w_a_ls = [w1 * (time_x.unsqueeze(dim=1).expand(-1,w1.shape[0],-1)),  # 4*128*90
        #               w2 * (hid_1[:, :, tim].unsqueeze(dim=1).expand(-1,w2.shape[0],-1)), # 4*64*128
        #               w3[i, :] * (hid_2[:, :, tim])] # 4*(1*)64

        #     pos_w_a_ls = [return_pos_neg(w_a)[0] for w_a in w_a_ls]
        #     neg_w_a_ls = [return_pos_neg(w_a)[1] for w_a in w_a_ls]
        #     # print(pos_w_a_ls,neg_w_a_ls,sep="/n")

        #     # print((pos_b1/torch.count_nonzero(pos_w_a_ls[0],dim=1)).unsqueeze(dim=-1).expand_as(pos_w_a_ls[0]).shape)
        #     pos_w_a_ls[0] = add_regular_faster(pos_w_a_ls[0],pos_b1,"pos",epsilon)
        #     # print(pos_w_a_ls[0])
        #     pos_w_a_ls[1] = add_regular_faster(pos_w_a_ls[1],pos_b2,"pos",epsilon)
        #     pos_w_a_ls[2] = add_regular_faster(pos_w_a_ls[2],pos_b3[i],"pos",epsilon)
        #     # print(pos_w_a_ls[2].shape)
        #     neg_w_a_ls[0] = add_regular_faster(neg_w_a_ls[0],neg_b1,"neg",epsilon)
        #     neg_w_a_ls[1] = add_regular_faster(neg_w_a_ls[1],neg_b2,"neg",epsilon)
        #     neg_w_a_ls[2] = add_regular_faster(neg_w_a_ls[2],neg_b3[i],"neg",epsilon)



        #     sigma_pos_w_a_ls = [pos_w_a_ls[0].sum(dim=-1),
        #                         pos_w_a_ls[1].sum(dim=-1),
        #                         pos_w_a_ls[2].sum(dim=-1)]
        #     sigma_pos_w_a_ls = [torch.where(e==0,1,e) for e in sigma_pos_w_a_ls]
        #     sigma_neg_w_a_ls = [neg_w_a_ls[0].sum(dim=-1),
        #                         neg_w_a_ls[1].sum(dim=-1),
        #                         neg_w_a_ls[2].sum(dim=-1)]
        #     sigma_neg_w_a_ls = [torch.where(e==0,1,e) for e in sigma_neg_w_a_ls]

        #     rlp_score = val
        #     for j in range(2, -1, -1):  # num of hidden layer
        #         # 从后往前计算relevance score
        #         # message_between = []
        #         # sigma_j = sigma_w_a_ls[j]
        #         pos_ratio_matrix = pos_w_a_ls[j] / (sigma_pos_w_a_ls[j].unsqueeze(dim=-1).expand_as(pos_w_a_ls[j]))
        #         neg_ratio_matrix = neg_w_a_ls[j] / (sigma_neg_w_a_ls[j].unsqueeze(dim=-1).expand_as(neg_w_a_ls[j]))
        #         # print(pos_ratio_matrix.shape,neg_ratio_matrix.shape)
        #         if j == 2:
        #             rlp_score = (rlp_score.unsqueeze(dim=-1).expand_as(pos_ratio_matrix) * 
        #                         (alpha * pos_ratio_matrix + beta * neg_ratio_matrix))
        #             # rlp_score = last_linear_wij * rlp_score
        #         else:
        #             rlp_score = (rlp_score.unsqueeze(dim=-1).expand_as(pos_ratio_matrix) *
        #                         (alpha * pos_ratio_matrix + beta * neg_ratio_matrix)).sum(dim=1)
        #     lv_ls.append(rlp_score / val.unsqueeze(dim=-1).expand_as(rlp_score))
        # avg_score = torch.stack(lv_ls).mean(dim=0) # 4*90
        S_all_time_score_ls.append(S_avg_score) #50*4*90
        Z_all_time_score_ls.append(Z_avg_score)
    S_result = torch.stack(S_all_time_score_ls).mean(dim=0) #4*90
    Z_result = torch.stack(Z_all_time_score_ls).mean(dim=0) #4*90
    return S_result, Z_result#,lv_ls


def getTrafficData(data_dir):
    f = h5py.File(data_dir,'r')
    data = pd.DataFrame(f['df']['block0_values'][:],columns = f['df']['axis0'][:],index = f['df']['axis1'][:])
    data.columns = [str(e, encoding='utf-8') for e in data.columns]
    data.columns = [int(e) for e in data.columns]
    return data

def calculate_distance(coord1, coord2):
    """
    计算两个经纬度坐标之间的距离

    :param coord1: 第一个坐标，格式为 (纬度, 经度)
    :param coord2: 第二个坐标，格式为 (纬度, 经度)
    :return: 距离（单位：千米）
    """
    distance = geodesic(coord1, coord2).kilometers
    return distance

def findTargetTrafficData(data,location_dir,target_sensor_id):
    # target_sensor_id:int
    location = pd.read_csv(location_dir) # all_number
    t = (location.loc[location['sensor_id'] == target_sensor_id, 'latitude'].values.tolist()[0],
         location.loc[location['sensor_id'] == target_sensor_id, 'longitude'].values.tolist()[0])
    location['distance'] = location.apply(lambda x:calculate_distance(t,(x['latitude'],x['longitude'])),axis=1)
    neighbor_location = location.sort_values(by='distance',ascending=True).head(55)
    neighbor_variables = neighbor_location['sensor_id'].values # int
    t_data = data.loc[:,neighbor_variables]
    return t_data

def res_to_excel(res_pd, loc_dir, type):
    location = pd.read_csv(loc_dir)
    location = location.set_index("sensor_id")
    sort_location = location.loc[res_pd.index,:]
    sort_location = sort_location.reset_index()
    sort_location.index = range(1,55+1)
    sort_location.columns = ["sensor_id","index","latitude","longitude"]
    sort_location = sort_location[["sensor_id","index","longitude","latitude"]]
    sort_location.to_excel("../LRP/res_"+type+"_neighbor55.xlsx")
    # sort_location.head(30).to_excel("res_" + type + "_top30.xlsx")


def AllsamplesLRPscore(model, data, data_index,select_idxs, config, device,alpha=2, beta=-1):
    # LRPscore_u_ls = np.zeros(config.INPUT_DIM)
    LRPscore_s_ls = np.zeros(config.INPUT_DIM)
    LRPscore_z_ls = np.zeros(config.INPUT_DIM)
    model.eval()
    model_named_parameters = [x for x in model.named_parameters()]
    train_len = config.TRAIN_LEN
    batch_size = config.BATCH_SIZE
    sample_num = len(select_idxs)

    for end_ind in select_idxs:
        d = data[:,end_ind-train_len:end_ind]
        # d.to(device)
        # print(device)
        # print(d.device)
        inp = torch.stack([d]*batch_size).to(device)
        out = model(inp)
        # print(len(out))
        with torch.no_grad():
            # u = out[9][0]  # 32*50
            z = out[4][0]  # 32*50
            s = out[5][0] # 32*50
            hid_z1, hid_z2, hid_z3, hid_s1, hid_s2, hid_s3 = out[-6:]
            # hid_u1:128*50, hid_u2:64*50, hid_u3:32*50
            # print(d.device)
            # print(u.device)
            # res_u = calculate_lrp_3layers_alphabeta(model_named_parameters, d, [hid_u1, hid_u2], u, "u", config, device,
            #                                         alpha, beta)
            res_s = calculate_lrp_3layers_alphabeta(model_named_parameters, d, [hid_s1, hid_s2], s, "s", config, device,
                                                    alpha, beta)
            res_z = calculate_lrp_3layers_alphabeta(model_named_parameters, d, [hid_z1, hid_z2], z, "z", config, device,
                                                    alpha, beta)
            # LRPscore_u_ls += res_u.cpu().numpy()
            LRPscore_s_ls += res_s.cpu().numpy()
            LRPscore_z_ls += res_z.cpu().numpy()

    # LRPscore_u_ls = LRPscore_u_ls / sample_num
    LRPscore_s_ls = LRPscore_s_ls / sample_num
    LRPscore_z_ls = LRPscore_z_ls / sample_num
    # LRPscore_u_pd = pd.DataFrame(LRPscore_u_ls,index=data_index,columns=["U_RC"]).sort_values("U_RC",ascending=False)
    LRPscore_s_pd = pd.DataFrame(LRPscore_s_ls, index=data_index, columns=["S_RC"]).sort_values("S_RC", ascending=False)
    LRPscore_z_pd = pd.DataFrame(LRPscore_z_ls, index=data_index, columns=["Z_RC"]).sort_values("Z_RC", ascending=False)
    return LRPscore_s_pd, LRPscore_z_pd

def AllsamplesLRPscore_faster(model, data, data_index,select_idxs, config, device,alpha=2, beta=-1):
    # LRPscore_u_ls = np.zeros(config.INPUT_DIM)
    LRPscore_s_ls = []
    LRPscore_z_ls = []
    model.eval()
    model_named_parameters = [x for x in model.named_parameters()]
    train_len = config.TRAIN_LEN
    # batch_size = config.BATCH_SIZE
    sample_num = len(select_idxs) // config.BATCH_SIZE

    with torch.no_grad():
        for i in range(sample_num):
            sample_ls = []
            for j in range(config.BATCH_SIZE):
                end_ind = select_idxs[i*config.BATCH_SIZE+j]
                sample_ls.append(data[:, end_ind-config.TRAIN_LEN:end_ind])
            # d = data[:,end_ind-train_len:end_ind]
            # d.to(device)
            # print(device)
            # print(d.device)
            # print(len(out))
            inp = torch.stack(sample_ls).to(device)
            # print(inp./shape)
            out = model(inp)
            # u = out[9][0]  # 32*50
            z = out[4]  # 4*32*50
            s = out[5] # 4*32*50
            hid_z1, hid_z2, hid_z3, hid_s1, hid_s2, hid_s3 = out[-6:]
            # hid_u1:128*50, hid_u2:64*50, hid_u3:32*50
            # print(d.device)
            # print(u.device)
            # res_u = calculate_lrp_3layers_alphabeta(model_named_parameters, d, [hid_u1, hid_u2], u, "u", config, device,
            #                                         alpha, beta)
            # res_s = calculate_lrp_3layers_alphabeta(model_named_parameters, d, [hid_s1, hid_s2], s, "s", config, device,
            #                                         alpha, beta)
            # res_z = calculate_lrp_3layers_alphabeta(model_named_parameters, d, [hid_z1, hid_z2], z, "z", config, device,
            #                                         alpha, beta)
            res_s, res_z = calculate_lrp_3layers_alphabeta_faster(model_named_parameters,inp, [hid_s1, hid_s2], 
                                                                  [hid_z1, hid_z2], s, z, config, device,
                                                                  alpha, beta)
            # LRPscore_u_ls += res_u.cpu().numpy()
            LRPscore_s_ls.append(res_s)
            LRPscore_z_ls.append(res_z)
            # LRPscore_s_ls += res_s.cpu().numpy()
            # LRPscore_z_ls += res_z.cpu().numpy()

        LRPscore_s = torch.stack(LRPscore_s_ls).reshape(config.EPOCH_SAMPLES, config.INPUT_DIM).mean(dim=0)
        LRPscore_z = torch.stack(LRPscore_z_ls).reshape(config.EPOCH_SAMPLES, config.INPUT_DIM).mean(dim=0)
        # LRPscore_s_ls = LRPscore_s_ls / sample_num
        # LRPscore_z_ls = LRPscore_z_ls / sample_num
        # LRPscore_u_pd = pd.DataFrame(LRPscore_u_ls,index=data_index,columns=["U_RC"]).sort_values("U_RC",ascending=False)
        LRPscore_s_pd = pd.DataFrame(LRPscore_s.cpu().numpy(), index=data_index, columns=["S_RC"]).sort_values("S_RC", ascending=False)
        LRPscore_z_pd = pd.DataFrame(LRPscore_z.cpu().numpy(), index=data_index, columns=["Z_RC"]).sort_values("Z_RC", ascending=False)
    return LRPscore_s_pd, LRPscore_z_pd


def normalize(data):
    # data: numpy 2D
    length = len(data)
    avg = np.tile(np.mean(data, axis=0), (length, 1))
    std = np.tile(np.std(data,axis=0), (length, 1))
    return (data - avg) / std



if __name__ == "__main__":
    
    config = Myconfig()

    os.system("python train2.py")
    print("end train2.py")

    device = config.device
    print(device)
    if config.DATA_NAME == "lorenz96":
        data_dir = "../data_files/lorenz96/lorenz96_n=60_F=5_dt=0.03_skipTimeNum=10000_timeRange=(0,3000).csv"
        data = pd.read_csv(data_dir,index_col=0).values.astype(np.float32)
    elif config.DATA_NAME == "power_grid":
        data_dir = "/home/csh/CReP/self_supervised_two_parts/data_files/power_grid/adjusted_power_grid.csv"
        data = pd.read_csv(data_dir,index_col=0).values.astype(np.float32)
    else:
        data_dir = "/home/csh/CReP/self_supervised_two_parts/data_files/dream4/dream4ts1_interval1_noexpnoise_noise0.01_sma5.csv"
        data = pd.read_csv(data_dir).values.astype(np.float32)
        data = data[100:,:]
    # data_dir = "../data_files/lorenz96/lorenz96_n=60_F=5_dt=0.03_skipTimeNum=10000_timeRange=(0,3000).csv"
    # data = pd.read_csv(data_dir,index_col=0).values.astype(np.float32)
    # data_dir = "/home/csh/CReP/self_supervised_two_parts/data_files/dream4/dream4ts1_interval1_noexpnoise_noise0.01_sma5.csv"
    # data = pd.read_csv(data_dir).values.astype(np.float32)
    # data = data[100:,:]
    # data_dir = "/home/csh/CReP/self_supervised_two_parts/data_files/power_grid/adjusted_power_grid.csv"
    # data = pd.read_csv(data_dir,index_col=0).values.astype(np.float32)
    data = normalize(data)
    print(data.shape)
    # getDataSkipStep = config.TRAIN_LEN
    getDataSkipStep = config.getDataSkipStep

    epoch_samples = config.EPOCH_SAMPLES
    select_idxs = data_loader.get_select_idxs(data.shape[0], # sample num
                                              config.TRAIN_LEN, # 50
                                              config.EMBEDDING_LEN, # 16
                                              getDataSkipStep,
                                              epoch_samples) # 1000
    
    data = torch.tensor(data.T).to(device)


    
    # data_dir = '../../data_files/lorenz'
    # lorenz_data = data_loader.load_lorenz_data(data_dir, config.INPUT_DIM // 3, skip_time_num=10000, time_invariant=True, time=5000)
    # lorenz_data = lorenz_data[10000:,:]
    # lorenz_data = lorenz96.gen_L96_data(N = config.INPUT_DIM, F = 5, time_range=(0,3000), dt=0.03, skip_time_num = 10000)
    # lorenz_data = lorenz_data.astype(np.float32)
    # # lorenz_data = torch.tensor(lorenz_data.T)

    # select_idxs = data_loader.get_select_idxs(lorenz_data.shape[0],  # 248000
    #                                              config.TRAIN_LEN,  # 50
    #                                              config.EMBEDDING_LEN,  # 16
    #                                              epoch_samples)  # 1000
    # data = hdf5storage.loadmat("../STCN_datasets/hk/hk_data_v1.mat")
    # lorenz_data = torch.tensor(data['data'], dtype=torch.float).numpy()
    # data = pd.read_csv("../STCN_datasets/jp_covid/japan_covid_data.csv")
    # lorenz_data = data.iloc[:, 1:].values
    # lorenz_data = lorenz_data.astype(np.float32)
    # lorenz_data = torch.tensor(lorenz_data.T)
    # data = pd.read_csv("../.iloc[:,1:].values,dtype=torch.float).numpy()

    # data_dir = "D:/研究生/学习/时间序列预测2023暑假/STCN_datasets/traffic/metr-la.h5"
    # traffic_data_pd = getTrafficData(data_dir)
    # # target_sensor_id = 769430
    # loc_dir = "D:/研究生/学习/时间序列预测2023暑假/STICM-main/datasets/traffic/graph_sensor_locations.csv"
    # target_sensor_id = traffic_data_pd.columns[config.TARGET_IDX]
    # target_neighbor_sensor = findTargetTrafficData(traffic_data_pd, loc_dir, target_sensor_id).columns.values
    # lorenz_data = torch.tensor(traffic_data_pd.values, dtype=torch.float).numpy()


    # data = pd.read_csv("../data_files/trajectory_new.csv")
    # data = data.iloc[:,1:]
    # lorenz_data = data.loc[data.index % 5 == 0].values  # 200000/5 = 40000
    # lorenz_data = torch.tensor(lorenz_data,dtype=torch.float).numpy()
    # lorenz_data = lorenz_data[5000:,:]
    # lorenz_data = normalize(lorenz_data)

    # data = pd.read_csv("../data_files/dream4ts1_interval1_noexpnoise_noise0.01_sma5.csv")
    # lorenz_data = data.values.astype(np.float32)
    # # lorenz_data = pd.DataFrame(lorenz_data)
    # # lorenz_data = lorenz_data.loc[lorenz_data.index % 3 == 0].values.astype(np.float32)
    # lorenz_data = lorenz_data[100:,:]
    # lorenz_data = normalize(lorenz_data)

    # epoch_samples = config.EPOCH_SAMPLES
    # select_idxs = data_loader.get_select_idxs_v2(lorenz_data.shape[0],  # 248000
    #                                              config.TRAIN_LEN,  # 50
    #                                              config.EMBEDDING_LEN,  # 16
    #                                              epoch_samples)  # 1000

    # select_idxs = select_idxs[:16]
    

    # model = torch.load("../result/power_v41.pth")
    model = Mymodel1(config)
    # model.load_state_dict(torch.load("../predict_results/lorenz96/y"+str(config.TARGET_IDX+1)+"("+str(config.TRAIN_LEN)+"pred"+str(config.EMBEDDING_LEN-1)+")("+str(config.residx)+").pth"))
    # model.load_state_dict(torch.load(
    #     "../predict_results/"+config.DATA_NAME+"/Ablation-G"+str(config.TARGET_IDX+1)+
    #            "("+str(config.LOSS_WEIGHTS['masked_embedding_loss'])+
    #            "|"+str(config.LOSS_WEIGHTS['future_consistency_loss'])+
    #            "|"+str(config.LOSS_WEIGHTS['reconstruction_loss_x'])+
    #            "|"+str(config.LOSS_WEIGHTS['orthogonal_loss'])+
    #            ")("+str(config.residx)+").pth"
    # ))
    model.load_state_dict(torch.load("../predict_results/"+config.DATA_NAME+"/G"+str(config.TARGET_IDX+1)+
               "("+str(config.residx)+").pth"))
    # model.load_state_dict(torch.load("../predict_results/"+config.DATA_NAME+"/Predlen-G"+str(config.TARGET_IDX+1)+
    #            "(50pred"+str(config.EMBEDDING_LEN-1)+")("+str(config.residx)+").pth"))
    # model.load_state_dict(torch.load("/home/csh/CReP/self_supervised_two_parts/result/dream4interval1_noexpnoise_noise0.01_m40l8_sma5(lap1_sample10000)_125.pth"))
    model.to(device)
    model.eval()

    # ind = range(0,120)
    # ind = []
    # for i in range(1, 15):
    #     if i == 1:
    #         ind.append("cardio")
    #     elif i == 2:
    #         ind.append("resp")
    #     elif i == 3:
    #         ind.append("no2")
    #     elif i == 4:
    #         ind.append("so2")
    #     elif i == 5:
    #         ind.append("rspar")
    #     elif i == 6:
    #         ind.append("o3")
    #     elif i == 7:
    #         ind.append("temp")
    #     elif i == 8:
    #         ind.append("hum")
    #     else:
    #         ind.append(str(i))
    # ind = [str(i) for i in range(1,config.INPUT_DIM+1)]
    ind = ["G"+str(i) for i in range(1,config.INPUT_DIM+1)]



    # ind = [str(i) for i in range(0,120)]
    # ind = ["G"+str(i) for i in range(1,51)]
    # ind = data.columns[1:]
    S_RC, Z_RC = AllsamplesLRPscore_faster(model, data, ind, select_idxs, config, device, alpha=2,beta=-1)
    S_RC.index = S_RC.index.values.astype("str")
    Z_RC.index = Z_RC.index.values.astype("str")
    # U_RC.index = U_RC.index.values.astype("str")
    RC = pd.concat([S_RC,Z_RC],axis=1)
    # print(RC.sort_values("U_RC",ascending=False).head(10))
    print(RC.sort_values("S_RC",ascending=False).head(11))
    print(RC.sort_values("Z_RC",ascending=False).head(11))
    RC.to_excel("../RS_results/"+config.DATA_NAME+"/G"+str(config.TARGET_IDX+1)+
               "("+str(config.residx)+").xlsx")
    # RC.to_excel("../RS_results/"+config.DATA_NAME+"/Predlen-G"+str(config.TARGET_IDX+1)+
    #            "(50pred"+str(config.EMBEDDING_LEN-1)+")("+str(config.residx)+").xlsx")
    # RC.to_excel("../RS_results/dream4/test125.xlsx")
    # RC.to_excel("../RS_results/lorenz96/y"+str(config.TARGET_IDX+1)+"("+str(config.TRAIN_LEN)+"pred"+str(config.EMBEDDING_LEN-1)+")("+str(config.residx)+").xlsx")
    # RC.to_excel("../RS_results/"+config.DATA_NAME+"/Ablation-G"+str(config.TARGET_IDX+1)+
    #            "("+str(config.LOSS_WEIGHTS['masked_embedding_loss'])+
    #            "|"+str(config.LOSS_WEIGHTS['future_consistency_loss'])+
    #            "|"+str(config.LOSS_WEIGHTS['reconstruction_loss_x'])+
    #            "|"+str(config.LOSS_WEIGHTS['orthogonal_loss'])+
    #            ")("+str(config.residx)+").xlsx")

