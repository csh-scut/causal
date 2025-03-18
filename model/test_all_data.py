import sys
sys.path.append("..")
import os
from scipy.stats import mannwhitneyu
from data_process import data_loader, lorenz96
from model.config import Myconfig
from model.train2 import DataGenerator
from model.unit import Mymodel1
import torch
import pandas as pd
import h5py
from geopy.distance import geodesic

import matplotlib.pyplot as plt
import numpy as np
import hdf5storage
from model.loss import RMSELoss, PCCLoss
import warnings
warnings.filterwarnings("ignore")

def normalize(data):
    # data: numpy 2D
    length = len(data)
    avg = np.tile(np.mean(data, axis=0), (length, 1))
    std = np.tile(np.std(data,axis=0), (length, 1))
    return (data - avg) / std

def normalize(data):
    # data: numpy 2D
    length = len(data)
    avg = np.tile(np.mean(data, axis=0), (length, 1))
    std = np.tile(np.std(data,axis=0), (length, 1))
    return (data - avg) / std

def normalize_v2(y, avg, std):
    # data: numpy 2D

    return (y - avg) / std

def separate_variables(scores,res,type,file,thresh_percentile):
    percentiles = np.percentile(scores, thresh_percentile)
    # print("80th percentile (20%):", percentiles[0])
    # print("85th percentile (15%):", percentiles[1])
    # print("90th percentile (10%):", percentiles[2])
    # print("95th percentile (5%):", percentiles[3])
    print("begin to select the best threshold:--->",end="")
    thresholds = []
    for ele in percentiles:
        thresholds.append(ele)
    results = []
    # p_results = []
    for threshold in thresholds:
        # 按阈值分组
        group1 = scores[scores <= threshold]
        group2 = scores[scores > threshold]

        # 确保每组都有足够的数据
        if len(group1) > 0 and len(group2) > 0:
            # 执行 Mann-Whitney U 检验
            u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            results.append((threshold, u_stat, p_value))
            # p_results.append(p_value)
    
    # print(results)

    if max(results, key=lambda x: x[2])[2] < 0.05:
        allthreshusable = True
        res.loc[file,"allthreshusable"] = True
    else:
        allthreshusable = False
        res.loc[file,"allthreshusable"] = False
    # 找到具有最小 p 值的阈值
    best_threshold, best_u_stat, best_p_value = min(results, key=lambda x: x[2])
    
    # print(f"Best threshold: {best_threshold}")
    # print(f"U statistic: {best_u_stat}")
    # print(f"P value: {best_p_value}")

    if best_p_value < 0.05:
        print("statistically significant.")
        res.loc[file,"SIG of "+type] = True

    else:
        print("not statistically significant.")
        res.loc[file,"SIG of "+type] = False
        
    below_best_threshold_num = (scores < best_threshold).sum()
    above_best_threshold_num = (scores >= best_threshold).sum()
    return best_threshold, below_best_threshold_num, above_best_threshold_num, allthreshusable, res

def causalMetricforLorenz(resFile, config, res, threshs,target_start_end): 
    file = resFile[:-5] + ".pth"
    fullfile = "../RS_results/lorenz96/" + resFile
    rc = pd.read_excel(fullfile,index_col=0)
    # target = config.TARGET_IDX+1
    target = int(resFile[target_start_end[0]:target_start_end[1]])
    rc.drop("G"+str(target),inplace=True)
    label = pd.DataFrame(np.zeros(((config.INPUT_DIM-1)*2,3)),index=range(1,(config.INPUT_DIM-1)*2+1),columns=["true","p_value","binary"],dtype=int)
    ind = []
    for i in range(1,config.INPUT_DIM+1):
        if i == target:
            continue
        else:
            ind.append("G"+str(target)+"->G"+str(i))
            ind.append("G"+str(i)+"->G"+str(target))
    label.index = ind
    tmpcauses = [target-1,target+1,target-2]
    tmpeffects = [target-1,target+1,target+2]
    causes = []
    effects = []
    for i in range(3):
        c = tmpcauses[i]
        if c < 1:
            causes.append(config.INPUT_DIM+c)
        elif c > config.INPUT_DIM:
            causes.append(c-config.INPUT_DIM)
        else:
            causes.append(c)
        e = tmpeffects[i]
        if e < 1:
            effects.append(config.INPUT_DIM+e)
        elif e > config.INPUT_DIM:
            effects.append(e-config.INPUT_DIM)
        else:
            effects.append(e)

    for i in range(3):
        label.loc["G"+str(causes[i])+"->G"+str(target),"true"]=1
        label.loc["G"+str(target)+"->G"+str(effects[i]),"true"]=1

    # threshs = [80,85,90]
    # threshs = [80,85,90,95]
    s_scores = rc["S_RC"].values
    s_v_value,not_s_v_num,s_v_num,s_allthreshusable,res = separate_variables(s_scores, res, "cause",file,threshs)
    res.loc[file,"inferred cause"] = ", ".join(rc.sort_values("S_RC",ascending=False).index[:s_v_num].values)
    # print("inferred cause: ",rc.sort_values("S_RC",ascending=False).index[:s_v_num])
    z_scores = rc["Z_RC"].values
    z_v_value,not_z_v_num,z_v_num,z_allthreshusable,res = separate_variables(z_scores, res, "effect",file,threshs)
    res.loc[file,"inferred effect"] = ", ".join(rc.sort_values("Z_RC",ascending=False).index[:z_v_num].values)
    # print("inferred effect: ",rc.sort_values("Z_RC",ascending=False).index[:z_v_num])

    # if s_allthreshusable:
    #     s_per = np.percentile(s_scores, threshs)[-1]
    #     s_above_threshold_num = (s_scores >= s_per).sum()
    # else:
    s_per = s_v_value
    s_above_threshold_num = s_v_num
    
    # if z_allthreshusable:
    #     z_per = np.percentile(z_scores, threshs)[-1]
    #     z_above_threshold_num = (z_scores >= z_per).sum()
    # else:
    z_per = z_v_value
    z_above_threshold_num = z_v_num

    for c in rc.sort_values("S_RC",ascending=False).index[:s_above_threshold_num]:
        label.loc[c+"->G"+str(target),"p_value"]=1
    for e in rc.sort_values("Z_RC",ascending=False).index[:z_above_threshold_num]:
        label.loc["G"+str(target)+"->"+e,"p_value"]=1
    
    res.loc[file,"TP"] = label.loc[label["true"]==1,"p_value"].mean()
    res.loc[file,"FP"] = label.loc[label["true"]==0,"p_value"].mean()

    return res
            # ["true"]==label["p_value"]).values.sum()/label["p_value"].sum()
    

def causalMetricforPowergrid(resFile, config, res, threshs,target_start_end): 
    file = resFile[:-5] + ".pth"
    fullfile = "../RS_results/power_grid/" + resFile
    rc = pd.read_excel(fullfile,index_col=0)
    # target = config.TARGET_IDX+1
    target = int(resFile[target_start_end[0]:target_start_end[1]])
    rc.drop("G"+str(target),inplace=True)
    label = pd.DataFrame(np.zeros(((config.INPUT_DIM-1)*2,3)),index=range(1,(config.INPUT_DIM-1)*2+1),columns=["true","p_value","binary"],dtype=int)
    ind = []
    for i in range(1,config.INPUT_DIM+1):
        if i == target:
            continue
        else:
            ind.append("G"+str(target)+"->G"+str(i))
            ind.append("G"+str(i)+"->G"+str(target))
    label.index = ind
    truenet = pd.read_csv("../data_files/power_grid/E_cause_36to33.csv",index_col=0).values
    causes = []
    effects = []
    for edge in truenet:
        if edge[0]+1 == target:
            effects.append(edge[1]+1)
        if edge[1]+1 == target:
            causes.append(edge[0]+1)
    res.loc[file,"true cause"] = ",".join(["G"+str(i) for i in causes])
    res.loc[file,"true effect"] = ",".join(["G"+str(i) for i in effects])

    for i in range(len(causes)):
        label.loc["G"+str(causes[i])+"->G"+str(target),"true"]=1
    for i in range(len(effects)):
        label.loc["G"+str(target)+"->G"+str(effects[i]),"true"]=1

    # threshs = [80,85,90]
    # threshs = [80,85,90,95]
    s_scores = rc["S_RC"].values
    s_v_value,not_s_v_num,s_v_num,s_allthreshusable,res = separate_variables(s_scores, res, "cause",file,threshs)
    res.loc[file,"inferred cause"] = ", ".join(rc.sort_values("S_RC",ascending=False).index[:s_v_num].values)
    # print("inferred cause: ",rc.sort_values("S_RC",ascending=False).index[:s_v_num])
    z_scores = rc["Z_RC"].values
    z_v_value,not_z_v_num,z_v_num,z_allthreshusable,res = separate_variables(z_scores, res, "effect",file,threshs)
    res.loc[file,"inferred effect"] = ", ".join(rc.sort_values("Z_RC",ascending=False).index[:z_v_num].values)
    # print("inferred effect: ",rc.sort_values("Z_RC",ascending=False).index[:z_v_num])

    # if s_allthreshusable:
    #     s_per = np.percentile(s_scores, threshs)[-1]
    #     s_above_threshold_num = (s_scores >= s_per).sum()
    # else:
    s_per = s_v_value
    s_above_threshold_num = s_v_num

    # if z_allthreshusable:
    #     z_per = np.percentile(z_scores, threshs)[-1]
    #     z_above_threshold_num = (z_scores >= z_per).sum()
    # else:
    z_per = z_v_value
    z_above_threshold_num = z_v_num

    for c in rc.sort_values("S_RC",ascending=False).index[:s_above_threshold_num]:
        label.loc[c+"->G"+str(target),"p_value"]=1
    for e in rc.sort_values("Z_RC",ascending=False).index[:z_above_threshold_num]:
        label.loc["G"+str(target)+"->"+e,"p_value"]=1
    
    res.loc[file,"TP"] = label.loc[label["true"]==1,"p_value"].mean()
    res.loc[file,"FP"] = label.loc[label["true"]==0,"p_value"].mean()

    return res

def causalMetricforDream4(resFile, config, res, threshs): 
    file = resFile[:-5] + ".pth"
    fullfile = "../RS_results/dream4/" + resFile
    rc = pd.read_excel(fullfile,index_col=0)
    # target = config.TARGET_IDX+1
    target = int(resFile[1:-8])
    rc.drop("G"+str(target),inplace=True)
    label = pd.DataFrame(np.zeros(((config.INPUT_DIM-1)*2,3)),index=range(1,(config.INPUT_DIM-1)*2+1),columns=["true","p_value","binary"],dtype=int)
    ind = []
    for i in range(1,config.INPUT_DIM+1):
        if i == target:
            continue
        else:
            ind.append("G"+str(target)+"->G"+str(i))
            ind.append("G"+str(i)+"->G"+str(target))
    label.index = ind
    truenet = pd.read_csv('../data_files/dream4/InSilicoSize50-Yeast1.tsv', delimiter='\t',header=None)
    truenet.columns=["from","to","edge"]
    truenet = truenet.loc[truenet['edge']==1,["from","to"]].values
    causes = []
    effects = []
    for edge in truenet:
        if int(edge[0][1:]) == target:
            effects.append(int(edge[1][1:]))
        if int(edge[1][1:]) == target:
            causes.append(int(edge[0][1:]))

    for i in range(len(causes)):
        label.loc["G"+str(causes[i])+"->G"+str(target),"true"]=1
    for i in range(len(effects)):
        label.loc["G"+str(target)+"->G"+str(effects[i]),"true"]=1

    # threshs = [80,85,90]
    # threshs = [80,85,90,95]
    s_scores = rc["S_RC"].values
    s_v_value,not_s_v_num,s_v_num,s_allthreshusable,res = separate_variables(s_scores, res, "cause",file,threshs)
    res.loc[file,"inferred cause"] = ", ".join(rc.sort_values("S_RC",ascending=False).index[:s_v_num].values)
    # print("inferred cause: ",rc.sort_values("S_RC",ascending=False).index[:s_v_num])
    z_scores = rc["Z_RC"].values
    z_v_value,not_z_v_num,z_v_num,z_allthreshusable,res = separate_variables(z_scores, res, "effect",file,threshs)
    res.loc[file,"inferred effect"] = ", ".join(rc.sort_values("Z_RC",ascending=False).index[:z_v_num].values)
    # print("inferred effect: ",rc.sort_values("Z_RC",ascending=False).index[:z_v_num])

    # if s_allthreshusable:
    #     s_per = np.percentile(s_scores, threshs)[-1]
    #     s_above_threshold_num = (s_scores >= s_per).sum()
    # else:
    s_per = s_v_value
    s_above_threshold_num = s_v_num
    
    # if z_allthreshusable:
    #     z_per = np.percentile(z_scores, threshs)[-1]
    #     z_above_threshold_num = (z_scores >= z_per).sum()
    # else:
    z_per = z_v_value
    z_above_threshold_num = z_v_num

    for c in rc.sort_values("S_RC",ascending=False).index[:s_above_threshold_num]:
        label.loc[c+"->G"+str(target),"p_value"]=1
    for e in rc.sort_values("Z_RC",ascending=False).index[:z_above_threshold_num]:
        label.loc["G"+str(target)+"->"+e,"p_value"]=1
    
    res.loc[file,"TP"] = label.loc[label["true"]==1,"p_value"].mean()
    res.loc[file,"FP"] = label.loc[label["true"]==0,"p_value"].mean()

    return res
        


    
    


if __name__ == '__main__':
    config = Myconfig()
    device = config.device
    # data_dir = '../../data_files/lorenz'
    # lorenz_data = data_loader.load_lorenz_data(data_dir, config.INPUT_DIM // 3, skip_time_num=2000, time_invariant=True, time=5000)
    # lorenz_data = lorenz96.gen_L96_data(N = config.INPUT_DIM, F = 5, time_range=(0,3000), dt=0.03, skip_time_num = 10000)
    # lorenz_data = lorenz_data.astype(np.float32)

    # data_dir = "../data_files/lorenz96/lorenz96_n=60_F=5_dt=0.03_skipTimeNum=10000_timeRange=(0,3000).csv"
    # data = pd.read_csv(data_dir,index_col=0).values.astype(np.float32)
    data_dir = "/home/csh/CReP/self_supervised_two_parts/data_files/power_grid/adjusted_power_grid.csv"
    data = pd.read_csv(data_dir,index_col=0).values.astype(np.float32)
    # data_dir = "/home/csh/CReP/self_supervised_two_parts/data_files/dream4/dream4ts1_interval1_noexpnoise_noise0.01_sma5.csv"
    # data = pd.read_csv(data_dir).values.astype(np.float32)
    # data = data[100:,:]

    # print(lorenz_data.shape, lorenz_data)
    # data = pd.read_csv("../data_files/trajectory(34).csv")
    # data = data.iloc[:,1:]
    # print(data.shape)
    # lorenz_data = data.loc[data.index % 5 == 0].values  # 200000/5 = 40000
    # lorenz_data = torch.tensor(lorenz_data,dtype=torch.float).numpy()
    # data = pd.read_csv("../STCN_datasets/jp_covid/japan_covid_data.csv")
    # lorenz_data = data.iloc[:, 1:].values
    # data = hdf5storage.loadmat("../STCN_datasets/hk/hk_data_v1.mat")
    # lorenz_data = torch.tensor(data['data'], dtype=torch.float).numpy()
    # lorenz_data = lorenz_data.astype(np.float32)
    # print(lorenz_data.shape)
    # lorenz_data = lorenz_data[5000:,:]
    # lorenz_data = normalize(lorenz_data)
    # lorenz_data = torch.tensor(lorenz_data.T)
    # test_data = lorenz_data[:,config.EPOCH_SAMPLES*config.TRAIN_LEN:]

    # data = pd.read_csv("../data_files/trajectory_new.csv")
    # data = data.iloc[:,1:]
    # lorenz_data = data.loc[data.index % 5 == 0].values  # 200000/5 = 40000
    # lorenz_data = torch.tensor(lorenz_data,dtype=torch.float).numpy()
    # lorenz_data = lorenz_data[5000:,:]
    # lorenz_data = normalize(lorenz_data)

    # data = pd.read_csv("../data_files/dream4ts1_interval1_noexpnoise_noise0.01_sma5.csv")
    # lorenz_data = data.values.astype(np.float32)
    # lorenz_data = lorenz_data[100:,:]
    # lorenz_data = normalize(lorenz_data)
    # data = pd.DataFrame(data)
    # data = data.loc[data.index % 3 == 0].values.astype(np.float32)
    
    data = normalize(data)
    print(data.shape)
    getDataSkipStep = config.getDataSkipStep
    # getDataSkipStep = 1
#

    epoch_samples = config.EPOCH_SAMPLES
    select_idxs = data_loader.get_select_idxs(data.shape[0], # 248000
                                                config.TRAIN_LEN, # 50
                                                config.EMBEDDING_LEN, # 16
                                                getDataSkipStep,
                                                epoch_samples) # 1000
    # print(select_idxs)
    # select_idxs = select_idxs[8:]
    # print(len(select_idxs),select_idxs[-1])

    # test_idxs = [e+config.EPOCH_SAMPLES*config.TRAIN_LEN for e in select_idxs]
    
    threshs = [80,85,90,95]
    # model = torch.load("../result/lorenz96_y2(3).pth")
    predresfiles = os.listdir("../predict_results/"+config.DATA_NAME)
    # predresfiles = ["theta34(0).pth","theta34(1).pth","theta34(2).pth","theta34(3).pth"]
    # predresfiles = ["../predict_results/"]

    # validfile = ["G32(1).pth","G33(1).pth","G32(2).pth"]
    validfile = []
    for file in predresfiles:
        if "11" in file or "22" in file or "33" in file:
            validfile.append(file)
    # print(len(validfile))
    # validfile = ["y17(old_setting).pth","y21(old_setting).pth","y39(old_setting).pth","y29(old_setting).pth"]
    # validfile = ["G39(11).pth","G69(11).pth","G104(11).pth","G63(11).pth","G29(11).pth","G108(11).pth"]
    # validfile = ["y"+str(config.TARGET_IDX+1)+"(old_setting).pth"]
    res = pd.DataFrame(index=validfile,columns=["target","RMSE","PCC","SIG of cause","inferred cause","true cause","SIG of effect","inferred effect","true effect","allthreshusable","TP","FP"])
    for file in validfile:
        # if file == "G32(3).pth":
        #     continue
        print(file,end=", ")
        target_start_end = (1,-9)
        # if file[target_start_end[0]:target_start_end[1]] == "2":
        #     rcfile = "y2(original)"
        rcfile = file[:-4] + ".xlsx"
        # print(rcfile,target_start_end)
        fullfile = "../predict_results/"+config.DATA_NAME+"/" + file
        fullrcfile = "../RS_results/"+config.DATA_NAME+"/" + rcfile
        model = Mymodel1(config)
        model.load_state_dict(torch.load(fullfile))
        model.to(device)
        model.eval()
        test_loss_ls = []
        pcc_loss_ls = []
        # avg = np.mean(data[:,config.TARGET_IDX])
        # std = np.std(data[:,config.TARGET_IDX])
        # print(rcfile,rcfile[5:-18])
        # target = int(rcfile[5:-18])-1
        # target = int()
        
        test_loader = DataGenerator (target_idx=int(rcfile[target_start_end[0]:target_start_end[1]])-1,# 0target = 
                                #  target_idx = config.TARGET_IDX, # int(rcfile[5:-18])-1 
                                    train_len=config.TRAIN_LEN,# 50
                                    embedding_len=config.EMBEDDING_LEN,# 16
                                    select_idxs=select_idxs, # [50,100,...,50000]
                                    data_matrix=data, # 二维array数据
                                    batch_size=config.BATCH_SIZE, # 4
                                    shuffle=False)
        
        res.loc[file,"target"] = int(rcfile[target_start_end[0]:target_start_end[1]])
        # res.loc[file,"target"] = config.TARGET_IDX+1
        for batch in iter(test_loader):
            # iteration += 1
            # if (iteration > n_iterations) or (break_flag):
            #     continue
            # model.train()
            # Parse batch
            batched_input_data = batch['batched_input_data'].to(device)
            # print(batched_input_data.data)
            batched_target_embedding = batch['batched_target_embedding'].to(device)
            # print(batched_target_embedding.data)
            batched_target_y = batch['batched_target_y'].to(device)
            # print(model(batched_input_data).shape)
            predict_y = model(batched_input_data)[0]
            test_loss_ls.append(RMSELoss(predict_y,batched_target_y))
            pcc_loss_ls.append(PCCLoss(predict_y,batched_target_y))
        res.loc[file,"RMSE"] = (sum(test_loss_ls)/len(test_loss_ls)).detach().cpu().item()
        res.loc[file,"PCC"] = (sum(pcc_loss_ls)/len(pcc_loss_ls))


        # print(sum(test_loss_ls)/len(test_loss_ls),end=", ")
        # print(sum(pcc_loss_ls)/len(pcc_loss_ls),end="; ")
        if config.DATA_NAME == 'lorenz96':
            res = causalMetricforLorenz(rcfile,config,res,threshs,target_start_end)
        if config.DATA_NAME == 'power_grid':
            res = causalMetricforPowergrid(rcfile,config,res,threshs,target_start_end)
        if config.DATA_NAME == 'dream4':
            res = causalMetricforDream4(rcfile,config,res,threshs)
        # break

    # print(res)
    res.to_excel("../RS_results/"+config.DATA_NAME+"/all(FPR).xlsx")
            # print()
            # print()





    

    

    # model.load_state_dict(torch.load("../result/lorenz96_y2(3).pth"))
    # model.load_state_dict(torch.load("../result/dream4interval1_noexpnoise_noise0.01_m40l8_sma5(lap1_sample10000).pth"))
    # model.load_state_dict(torch.load("../result/ablation_COVID_con.pth"))
    # model.load_state_dict(torch.load("../result/ablation_power964_rec.pth"))
    # model.load_state_dict(torch.load("../result/ablation_power964_future.pth"))
    # model.load_state_dict(torch.load("../result/ablation_power964_ds.pth"))
    # model.load_state_dict(torch.load("../result/HK_464_unnorm_35.pth"))
    
    
    # for batch in iter(test_loader):
    #     # iteration += 1
    #     # if (iteration > n_iterations) or (break_flag):
    #     #     continue
    #     # model.train()
    #     # Parse batch
    #     batched_input_data = batch['batched_input_data'].to(device)
    #     batched_target_embedding = batch['batched_target_embedding'].to(device)
    #     batched_target_y = batch['batched_target_y'].to(device)
    #     # print(batched_input_data.requires_grad_)

    #     # optimizer.step()
    #     # print()
    #     # print()

    #     predict_y = model(batched_input_data)[0]

    #     # batched_target_y_ = normalize_v2(batched_target_y, avg, std)
    #     # predict_y_ = normalize_v2(predict_y, avg, std)
    #     # print("--------")
    #     # print(batched_target_y)
    #     # print(predict_y)
    #     test_loss_ls.append(RMSELoss(predict_y,batched_target_y))
    #     pcc_loss_ls.append(PCCLoss(predict_y,batched_target_y))

    # print(pd.DataFrame([e.item() for e in test_loss_ls],columns=["RMSE"]).sort_values("RMSE",ascending=False).head(10))
    # # print(pcc_loss_ls)
    # print(sum(test_loss_ls)/len(test_loss_ls))
    # print(sum(pcc_loss_ls)/len(pcc_loss_ls))
    # tmp_loss_ls = []
    # for e in test_loss_ls:
    #     # if e.item() > 1:
    #     #     continue
    #     tmp_loss_ls.append(e.item())

    # print(sum(tmp_loss_ls)/len(tmp_loss_ls))
    # print(sum(pcc_loss_ls)/len(pcc_loss_ls))
        



    # b = out[-1]