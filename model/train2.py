import sys
sys.path.append("..")
from data_process import data_loader, lorenz96
import pandas as pd
import datetime
import hdf5storage
from model.config import Myconfig
from geopy.distance import geodesic
from model.unit import Embedding2Y
import torch
import torch as t
import torch.nn as nn
from model.unit import Mymodel1
from model.loss import RMSELoss, MaskEmbeddingLoss, OrthogonalLoss, FutureConsistencyLoss
from torch import optim
import numpy as np
import time
import h5py
from captum.attr import LRP



class DataGenerator(object):
    def __init__(self, target_idx, train_len, embedding_len, select_idxs, data_matrix, batch_size, shuffle=True):
        """
        生成dropout的输入数据，以及embedding matrix、
        :param select_idxs: 时间点索引列表，多样本开始预测的时间点
        :param data_matrix: [time_len, input_dim]  所有数据
        :param batch_size: [train_len + ]
        """
        self.target_idx = target_idx
        self.select_idxs = select_idxs

        self.data_matrix = data_matrix
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_len = train_len
        self.embedding_len = embedding_len

    def __len__(self):
        # 按照batch，会有多少组数据
        return len(self.select_idxs) // self.batch_size

    def __iter__(self):
        # if self.shuffle:
        #     sample_idxs = np.random.choice(a=self.windows_sampling_idx,
        #                                    size=len(self.windows_sampling_idx), replace=False)
        # else:
        #     sample_idxs = self.windows_sampling_idx

        # assert len(sample_idxs)>0, 'Check the data as sample_idxs are empty'

        # 表示有多少批数据（batch=512）
        n_batches = int(np.ceil(len(self.select_idxs) / self.batch_size))
        # print(n_batches)
        # n_batches = int(np.ceil(len(sample_idxs) / self.batch_size)) # Must be multiple of batch_size for paralel gpu

        # sample_idxs是一个列表
        for idx in range(n_batches):
            # ws_idxs对应的是当前批（512个单位/窗口）的（窗口）索引：列表
            # ws_idxs = sample_idxs[(idx * self.batch_size) : (idx + 1) * self.batch_size]
            batch = self.__getitem__(idx)
            yield batch

    def __getitem__(self, item):  # item是一个数值
        batched_input_data = []
        batched_target_embedding = []
        batched_target_y = []
        # print(self.data_matrix.shape)

        for i in range(self.batch_size): # i=0,1,2,3

            t_idx = self.select_idxs[item*self.batch_size + i]  # 预测开始的时间点
            # m=50
            input_data = self.data_matrix[t_idx - self.train_len: t_idx] # 当前batch的某一组输入数据：50*90
            # print(input_data.shape)
            # total_y的长度：50+16-1
            total_y = self.data_matrix[t_idx - self.train_len: t_idx + self.embedding_len - 1, self.target_idx].copy()
            # print(self.data_matrix.shape)
            # if len(total_y)==42:
            #     print(item,i)
            #     print(t_idx)
            # print(total_y.shape)
            # print("before:",total_y)
            # print("before:")
            # total_y[self.train_len:] = 0  # 把未来时刻的数据设为0
            # print("after:",total_y)
            # print(total_y.shape)
            # 把total_y转换为embedding matrix
            target_embedding = []
            for i in range(self.train_len):  # 50
                target_embedding.append(total_y[i:i + self.embedding_len])
                # if
                # print(target_embedding[-1].shape)
            # print(target_embedding)
            target_embedding = np.stack(target_embedding)  # 默认竖着堆砌
            # target_embedding: 50*16

            # 未来时刻的真实值：一个batch有4组数据
            batched_target_y.append(total_y[- self.embedding_len + 1:])
            # batched_target_y.append(total_y)
            batched_input_data.append(input_data)
            batched_target_embedding.append(target_embedding)

        # 当前batch的输入（从整个lorenz_data中取），以及对应的输出
        batched_input_data = np.stack(batched_input_data) # dim:4,50,90
        # print(batched_input_data)
        batched_target_embedding = np.stack(batched_target_embedding) # dim:4,50,16
        # print(batched_target_embedding.shape)
        batched_target_y = np.stack(batched_target_y) # dim:4,16

        # return batched_input_data, [batched_target_embedding, batched_target_y, batched_input_data]
        return {'batched_input_data': torch.transpose(torch.tensor(batched_input_data),1,2),
                'batched_target_embedding': torch.tensor(batched_target_embedding),
                'batched_target_y':torch.tensor(batched_target_y)}

    def get_item(self, item):
        return self.__getitem__(item)

    # def on_epoch_end(self):
    #     if self.shuffle:
    #         np.random.shuffle(self.select_idxs)


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

def normalize(data):
    # data: numpy 2D
    length = len(data)
    avg = np.tile(np.mean(data, axis=0), (length, 1))
    std = np.tile(np.std(data,axis=0), (length, 1))
    return (data - avg) / std



### 训练



if __name__ == '__main__':


    config = Myconfig()
    print("learning_rate:",config.LR)
    print("batch_size:",config.BATCH_SIZE)
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




    # data_dir = '../../data_files/lorenz'
    # lorenz_data = data_loader.load_lorenz_data(data_dir, config.INPUT_DIM // 3,
    #                                            skip_time_num=10000, time_invariant=True, time=5000)
    # print(lorenz_data.dtype)
    # lorenz_data = lorenz96.gen_L96_data(N = config.INPUT_DIM, F = 5, time_range=(0,3000), dt=0.03, skip_time_num = 10000)
    # print(lorenz_data.dtype)
    # lorenz_data = lorenz_data.astype(np.float32)
    # lorenz_data = lorenz_data[10000:,:]
    #
    # # lorenz_data.shape:(248000,90) numpy二维对象
    # data = hdf5storage.loadmat("../STCN_datasets/hk/hk_data_v1.mat")
    # lorenz_data = torch.tensor(data['data'],dtype=torch.float).numpy()
    # data = pd.read_csv("../STCN_datasets/jp_covid/japan_covid_data.csv")
    # lorenz_data = torch.tensor(data.iloc[:,1:].values,dtype=torch.float).numpy()



    # data = pd.read_csv("../data_files/trajectory_new(2).csv")
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
    # select_idxs = data_loader.get_select_idxs_v2(lorenz_data.shape[0], # 248000
    #                                           config.TRAIN_LEN, # 50
    #                                           config.EMBEDDING_LEN, # 16
    #                                           epoch_samples) # 1000
    # select_idxs = np.append(select_idxs[:220],select_idxs[220:]+470-select_idxs[-1])
    # select_idxs[220:] + (470-select_id
    # select_idxs:[50,100,150,...,50000]

    # select_idxs = [config.TRAIN_LEN] * epoch_samples
    # print(select_idxs)


    time_stamp = datetime.datetime.now()


    # tf.keras.backend.clear_session()  # 清空GPU缓存

    train_loader = DataGenerator ( target_idx=config.TARGET_IDX,# 0
                                   train_len=config.TRAIN_LEN,# 50
                                   embedding_len=config.EMBEDDING_LEN,# 16
                                   select_idxs=select_idxs, # [50,100,...,50000]
                                   data_matrix=data, # 二维array数据
                                   batch_size=config.BATCH_SIZE, # 4
                                   shuffle=True)



    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = config.device
    model = Mymodel1(config)
    # model.load_state_dict(torch.load("../predict_results/dream4/G32(1).pth"))
    # model = torch.load( "../result/lorenz_y1.pth")
    model.to(device)
    # inputs = train_loader
    
    # print(model)#['model']

    # self.model = Mymodel1(self.block_list, self.config, self.embedding2y_layer).to(self.device)
    n_epoches = config.EPOCHES

    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30,40], gamma=0.1,last_epoch=-1)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.LR_DECAY, patience=6,min_lr=1e-6)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.N_LR_DECAY_STEPS,
    #                                          gamma=config.LR_DECAY)

    print('\n')
    print('=' * 30 + ' Start fitting ' + '=' * 30)

    start = time.time()
    # print(lorenz_data.type())
    for key in config.LOSS_WEIGHTS:
        if key == "masked_embedding_loss":
            w_mask_embedding = config.LOSS_WEIGHTS[key]
        elif key == 'future_consistency_loss':
            w_future = config.LOSS_WEIGHTS[key]
        elif key == 'reconstruction_loss_hidden':
            w_resconstruct_u = config.LOSS_WEIGHTS[key]
        elif key == 'reconstruction_loss_x':
            w_resconstruct_x = config.LOSS_WEIGHTS[key]
        elif key == 'consistency_loss':
            w_consistency = config.LOSS_WEIGHTS[key]
        elif key == 'orthogonal_loss':
            w_orthogonal = config.LOSS_WEIGHTS[key]
        else:
            continue
    epoch = 0
    epoches_loss_ls = []
    train_loss_ls = []
    # prediction_loss_ls = []
    # embedding_loss_ls = []
    future_loss_z_ls = []
    # future_loss_u_ls = []
    embedding_loss_z_ls = []
    # embedding_loss_u_ls = []
    # consistency_loss_ls = []
    orthogonal_loss_s_z_ls = []
    # orthogonal_loss_s_u_ls = []
    # orthogonal_loss_u_z_ls = []
    reconstruction_loss_s_ls = []
    reconstruction_loss_x_ls = []
    # reconstruction_loss_u_ls = []

    while (epoch < n_epoches):

        epoch += 1

        b_loss_ls = []
        # b_pred_loss_ls = []
        # b_embed_loss_ls = []
        b_future_loss_z_ls = []
        # b_future_loss_u_ls = []
        b_embed_loss_z_ls = []
        # b_embed_loss_u_ls = []
        # b_consistency_loss_ls = []
        b_rec_loss_x_ls = []
        # b_rec_loss_u_ls = []
        b_rec_loss_s_ls = []
        b_orthogonal_s_z_ls = []
        # b_orthogonal_s_u_ls = []
        # b_orthogonal_u_z_ls = []
        for batch in iter(train_loader):
            # iteration += 1
            # if (iteration > n_iterations) or (break_flag):
            #     continue

            model.train()
            # Parse batch
            batched_input_data = batch['batched_input_data'].to(device)
            batched_target_embedding = batch['batched_target_embedding'].to(device)
            batched_target_y = batch['batched_target_y'].to(device)
            # print(batched_input_data.requires_grad_)

            optimizer.zero_grad()
            # optimizer.step()
            # print()
            # print(batched_input_data.type())
            predict_y,embedding, rec_x_z,rec_x_s, z,s,s_hat,\
            hid_z1, hid_z2, hid_z3,hid_s1, hid_s2, hid_s3 = model(batched_input_data)
            # predict_y,embedding, rec_x_z,rec_x_s, z,s,s_hat = model(batched_input_data)

            orthogonal_s_z_loss = OrthogonalLoss(s,z)
            # orthogonal_s_u_loss = OrthogonalLoss(s, u)
            # orthogonal_u_z_loss = OrthogonalLoss(u, z)
            # predict_loss = RMSELoss(batched_target_y, predict_y)
            future_loss_z = FutureConsistencyLoss(embedding, config)
            # future_loss_u = FutureConsistencyLoss(embedding_u, config)
            # embedding_loss = MaskEmbeddingLoss(batched_target_embedding, embedding)
            embedding_loss_z = MaskEmbeddingLoss(batched_target_embedding, embedding, device)
            # embedding_loss_u = MaskEmbeddingLoss(batched_target_embedding, embedding_u)
            # consistency_loss = RMSELoss(embedding_u, embedding_z)
            # reconstruct_loss_u = RMSELoss(u,u_hat)
            reconstruct_loss_x = RMSELoss(batched_input_data, rec_x_z) + \
                                 RMSELoss(batched_input_data, rec_x_s)
            reconstruct_loss_s = RMSELoss(s,s_hat)

            # b_pred_loss_ls.append(predict_loss)
            # b_embed_loss_ls.append(embedding_loss)
            b_future_loss_z_ls.append(future_loss_z)
            # b_future_loss_u_ls.append(future_loss_u)
            b_embed_loss_z_ls.append(embedding_loss_z)
            # b_embed_loss_u_ls.append(embedding_loss_u)
            # b_consistency_loss_ls.append(consistency_loss)
            b_rec_loss_x_ls.append(reconstruct_loss_x)
            # b_rec_loss_u_ls.append(reconstruct_loss_u)
            b_rec_loss_s_ls.append(reconstruct_loss_s)
            b_orthogonal_s_z_ls.append(orthogonal_s_z_loss)
            # b_orthogonal_s_u_ls.append(orthogonal_s_u_loss)
            # b_orthogonal_u_z_ls.append(orthogonal_u_z_loss)


            batch_train_loss = w_mask_embedding * (embedding_loss_z) + \
                               w_resconstruct_x * (reconstruct_loss_x + reconstruct_loss_s) + \
                               w_future * (future_loss_z)+\
                               w_orthogonal * (orthogonal_s_z_loss)
                               # w_resconstruct_u * reconstruct_loss_u + \


            # print(batch_train_loss)
            b_loss_ls.append(batch_train_loss)

            # Protection if exploding gradients
            # if not np.isnan(float(training_loss)):
            batch_train_loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        avg_loss = sum(b_loss_ls) / len(b_loss_ls)
        # print(optimizer.param_groups)
        lr_scheduler.step(avg_loss)

        print("学习率", optimizer.param_groups[0]['lr'])

        # 一次 epoch 结束
        print("第" + str(epoch) + "个epoch:", end=" ")
        print(avg_loss)
        # print("++++++++++++++++++++++++++")
        epoches_loss_ls.append(avg_loss)
        # prediction_loss_ls.append(sum(b_pred_loss_ls) / len(b_pred_loss_ls))
        # embedding_loss_ls.append(sum(b_embed_loss_ls) / len(b_embed_loss_ls))
        embedding_loss_z_ls.append(sum(b_embed_loss_z_ls) / len(b_embed_loss_z_ls))
        # embedding_loss_u_ls.append(sum(b_embed_loss_u_ls) / len(b_embed_loss_u_ls))
        future_loss_z_ls.append(sum(b_future_loss_z_ls) / len(b_future_loss_z_ls))
        # future_loss_u_ls.append(sum(b_future_loss_u_ls) / len(b_future_loss_u_ls))
        # consistency_loss_ls.append(sum(b_consistency_loss_ls) / len(b_consistency_loss_ls))
        reconstruction_loss_x_ls.append(sum(b_rec_loss_x_ls) / len(b_rec_loss_x_ls))
        # reconstruction_loss_u_ls.append(sum(b_rec_loss_u_ls) / len(b_rec_loss_u_ls))
        reconstruction_loss_s_ls.append(sum(b_rec_loss_s_ls) / len(b_rec_loss_s_ls))
        orthogonal_loss_s_z_ls.append(sum(b_orthogonal_s_z_ls) / len(b_orthogonal_s_z_ls))
        # orthogonal_loss_s_u_ls.append(sum(b_orthogonal_s_u_ls) / len(b_orthogonal_s_u_ls))
        # orthogonal_loss_u_z_ls.append(sum(b_orthogonal_u_z_ls) / len(b_orthogonal_u_z_ls))


        train_loss_ls.append(b_loss_ls)
        if train_loader.shuffle:
            np.random.shuffle(train_loader.select_idxs)

    # train_loss = train_loss_ls
    # epoches_loss = epoches_loss_ls
    # pred_loss = prediction_loss_ls
    # embed_loss = embedding_loss_ls
    # pred_loss_z = prediction_loss_z_ls
    # pred_loss_u = prediction_loss_u_ls
    # embed_loss_z = embedding_loss_z_ls
    # embed_loss_u = embedding_loss_u_ls
    # consistency_loss = consistency_loss_ls
    # rec_loss_x = reconstruction_loss_x_ls
    # rec_loss_u = reconstruction_loss_u_ls

    string = 'Time: {:03.3f}'.format(time.time() - start)
    print(string)
    print('=' * 30 + '  End fitting  ' + '=' * 30)
    print('\n')

    # model.fit(inputs=train_loader)
    # torch.save(model.state_dict(), "../predict_results/"+config.DATA_NAME+"/Ablation-G"+str(config.TARGET_IDX+1)+
    #            "("+str(config.LOSS_WEIGHTS['masked_embedding_loss'])+
    #            "|"+str(config.LOSS_WEIGHTS['future_consistency_loss'])+
    #            "|"+str(config.LOSS_WEIGHTS['reconstruction_loss_x'])+
    #            "|"+str(config.LOSS_WEIGHTS['orthogonal_loss'])+
    #            ")("+str(config.residx)+").pth")

    torch.save(model.state_dict(), "../predict_results/"+config.DATA_NAME+"/G"+str(config.TARGET_IDX+1)+
               "("+str(config.residx)+").pth")
    # torch.save(model.state_dict(), "../predict_results/"+config.DATA_NAME+"/Predlen-G"+str(config.TARGET_IDX+1)+
    #            "(50pred"+str(config.EMBEDDING_LEN-1)+")("+str(config.residx)+").pth")
    el = [e.item() for e in epoches_loss_ls]

    # pl = [e.item() for e in prediction_loss_ls]
    # eml = [e.item() for e in embedding_loss_ls]
    # fu_u = [e.item() for e in future_loss_u_ls]
    # fu_z = [e.item() for e in future_loss_z_ls]
    # print(fu_z)
    # # eml_u = [e.item() for e in embedding_loss_u_ls]
    # eml_z = [e.item() for e in embedding_loss_z_ls]
    # print(eml_z)
    # rec_x = [e.item() for e in reconstruction_loss_x_ls]
    # print(rec_x)
    # rec_s = [e.item() for e in reconstruction_loss_s_ls]
    # print(rec_s)
    # orth = [e.item() for e in orthogonal_loss_s_z_ls]
    # print(orth)
    # ortho = [e.item() for e in orthogonal_loss_ls]
    # model1 = torch.load("D:/研究生/学习/时序预测（因果分析）/code/result/mymodel2(tanh).pth")
#

    # 测试
    # model.eval()

