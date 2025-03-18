'''
TemporalConvBlock:某一层TCN的结构
'''

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.init import xavier_uniform_ as xu_init
import numpy as np
import torch as t
from captum.attr import LRP

# def Embedding2Y(inputs, config):
#     # print(inputs.shape)
#     # embedding = torch.transpose(inputs, 1, 2).contiguous()
#
#     # print(embedding.shape)
#     target_y = embedding_2_predict_y(inputs, config, all_y=False)
#     # print(target_y.shape)
#     return target_y

class Embedding2Y(nn.Module):
    def __init__(self,config,**kwargs):
        super(Embedding2Y,self).__init__(**kwargs)
        self.config = config

    def forward(self, inputs,**kwargs):
        target_y = embedding_2_predict_y(inputs, self.config, all_y=False)
        return target_y


def get_same_idxs(sample_time_len, embedding_len):
    """
    获取embedding矩阵中斜线相等的元素的index
    :param sample_time_len:
    :param embedding_len:
    :return: [斜线num, [斜线元素num ,2]]
    """
    same_idxs = []
    if embedding_len > sample_time_len:
        for start_col_idx in range(1, embedding_len):
            idxs = []
            row_num = min(start_col_idx + 1, sample_time_len)
            for row_idx in range(row_num):
                idx = np.zeros((1, 2), dtype=np.int32)
                idx[0, 0] = row_idx
                idx[0, 1] = start_col_idx - row_idx
                idxs.append(idx)
            same_idxs.append(np.concatenate(idxs))

        # 下三角相等的
        for start_col_idx in range(embedding_len - sample_time_len + 1, embedding_len - 1):
            idx_count = embedding_len - start_col_idx
            idxs = []
            for i in range(idx_count):
                idx = np.zeros((1, 2), dtype=np.int32)
                idx[0, 0] = sample_time_len - 1 - i
                idx[0, 1] = start_col_idx + i
                idxs.append(idx)
            same_idxs.append(np.concatenate(idxs))
    else:
        for start_row_idx in range(1, sample_time_len):
            idxs = []
            col_num = min(start_row_idx+1, embedding_len)
            for col_idx in range(col_num):
                idx = np.zeros((1, 2), dtype=np.int32)
                idx[0, 1] = col_idx
                idx[0, 0] = start_row_idx - col_idx
                idxs.append(idx)
            same_idxs.append(np.concatenate(idxs))

        for start_row_idx in range(sample_time_len - embedding_len + 1, sample_time_len - 1):
            idx_count = sample_time_len - start_row_idx
            idxs = []
            for i in range(idx_count):
                idx = np.zeros((1, 2), dtype=np.int32)
                idx[0, 1] = embedding_len - 1 - i
                idx[0, 0] = start_row_idx + i
                idxs.append(idx)
            same_idxs.append(np.concatenate(idxs))

    return same_idxs

def gather_nd(params, indices):
    '''
    4D example
    params: tensor shaped [n_1, n_2, n_3, n_4] --> 4 dimensional
    indices: tensor shaped [m_1, m_2, m_3, m_4, 4] --> multidimensional list of 4D indices

    returns: tensor shaped [m_1, m_2, m_3, m_4]

    ND_example
    params: tensor shaped [n_1, ..., n_p] --> d-dimensional tensor
    indices: tensor shaped [m_1, ..., m_i, d] --> multidimensional list of d-dimensional indices

    returns: tensor shaped [m_1, ..., m_1]
    '''

    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1)  # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)

    # print(params.shape,idx)
    out = torch.take(params, idx)

    return out.view(out_shape)



def embedding_2_predict_y(delay_embedding, config, all_y=True):
    """

    :param delay_embedding: 此时已知
    :param config:
    :param batch_size:
    :param all_y: 是否返回预测出来的embedding矩阵里面全部的均值y
    :return:

    作用：把embeddingY矩阵中对应的预测值计算出来（平均）
    """
    # print(delay_embedding.shape)
    batch_size = config.BATCH_SIZE
    if all_y:
        same_idxs = get_same_idxs(config.TRAIN_LEN, config.EMBEDDING_LEN)
    else:
        same_idxs = get_same_idxs(config.TRAIN_LEN, config.EMBEDDING_LEN)[config.TRAIN_LEN - 1:]

    batch_predict_y = []
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    device = config.device
    for j in range(batch_size):
        predict_y = []
        if all_y:
            predict_y.append(delay_embedding[j, 0, 0])  # 把没有其他预测同一个Y的先放进来(time1)
        for i, same_idx in enumerate(same_idxs):
            same_y_s = gather_nd(delay_embedding[j,:,:], torch.tensor(same_idx).to(device))
            mean_y = torch.mean(same_y_s) # 1*1的tensor对象
            predict_y.append(mean_y)
        predict_y.append(delay_embedding[j, -1, -1])
        # predict_y：list 每个元素都是1*1的tendor对象
        predict_y = torch.stack(predict_y) # 拼接成一个1*n的tensor对象
        # print(predict_y.shape)
        batch_predict_y.append(predict_y)
    batch_predict_y = torch.stack(batch_predict_y)
    # print(batch_predict_y)
    return batch_predict_y

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, dilation, padding, activation='relu',
                 dropout = 0.2, last_norm = False):
        super(TemporalBlock, self).__init__()
        dict_activation = {'relu': nn.ReLU(),
                            'softplus': nn.Softplus(),
                            'tanh': nn.Tanh(),
                            'selu': nn.SELU(),
                            'lrelu': nn.LeakyReLU(),
                            'prelu': nn.PReLU(),
                            'sigmoid': nn.Sigmoid()}
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # nn.init.xavier_uniform_(self.conv1)
        # self.ln1 = nn.LayerNorm()
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.acti1 = dict_activation[activation]
        # self.bn1 = nn.BatchNorm1d(out_channels)
        # self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # nn.init.xavier_uniform_(self.conv2)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.acti2 = dict_activation[activation]
        # self.dropout2 = nn.Dropout(dropout)
        # self.bn2 = nn.BatchNorm1d(out_channels)

        if last_norm:
            self.net = nn.Sequential(self.conv1, self.chomp1,self.acti1,
                                 self.conv2, self.chomp2, self.acti2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.acti1,
                                    self.conv2, self.chomp2, self.bn2, self.acti2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.acti = dict_activation[activation]
        self.init_weights()

    def init_weights(self):
        # nn.init.xavier_uniform_(self.conv1.weight)
        # nn.init.xavier_uniform_(self.conv2.weight)
        #
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # print("TCNB：",self.dil, x.shape,self.conv1(x).shape,end=' ')

        # print(x.shape)
        out = self.net(x)
        # print(out.shape)
        # print("shape:",out.shape)
        res = x if self.downsample is None else self.downsample(x)
        return self.acti(out + res)
        # return out



class TCN_VAE_encoder(nn.Module):
    def __init__(self, input_dim, channels_list, dilation_list, kernel_size=3, activation='lrelu'):
        super(TCN_VAE_encoder, self).__init__()
        layers = []
        num_levels = len(channels_list)
        for i in range(num_levels):
            dilation_size = dilation_list[i]
            in_channels = input_dim if i == 0 else channels_list[i-1]
            # last_norm = True if i == num_levels-1 else False
            # if i == num_levels - 1:
            #     # 输出为mu和sigma，各16维
            #     out_channels = channels_list[i] * 2
            #     layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
            #                              padding=(kernel_size - 1) * dilation_size, activation=activation,last_norm=True)]
            # else:
            out_channels = channels_list[i]
            # print("第"+str(i+1)+"层:",in_channels,out_channels)
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, activation=activation,last_norm=last_norm)]

        self.network = nn.Sequential(*layers)
        # linear = nn.Linear(channels_list[-1],channels_list[-1])
        # dict_activation = {'relu': nn.ReLU(),
        #                    'softplus': nn.Softplus(),
        #                    'tanh': nn.Tanh(),
        #                    'selu': nn.SELU(),
        #                    'lrelu': nn.LeakyReLU(),
        #                    'prelu': nn.PReLU(),
        #                    'sigmoid': nn.Sigmoid()}
        #
        # activation = dict_activation[activation]
        # # print("nodes_num:",channels_list[-1])
        # bn = nn.BatchNorm1d(channels_list[-1],channels_list[-1])
        # # bn = nn.l
        # self.mu = nn.Sequential(linear,activation)
        # # self.m
        # self.sigma = nn.Sequential(linear,activation)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)  # 从标准正态分布中采样噪声
        z = mu + eps * std  # 重参数化技巧
        return z

    def forward(self, x):
        # print("TCN_AE:",x.shape)
        # x.shape:4*90*50
        # self.network(x).shape:4*32*50
        out = self.network(x)
        # print(out.shape)
        # mu = torch.transpose(self.mu(torch.transpose(out,1,2)),1,2)
        # logvar = torch.transpose(self.sigma(torch.transpose(out,1,2)),1,2)
        # z = self.reparameterize(mu, logvar)
        return out

class TCN_VAE_decoder(nn.Module):
    def __init__(self, input_dim, channels_list, dilation_list, kernel_size=3, activation='lrelu'):
        super(TCN_VAE_decoder, self).__init__()
        layers = []
        num_levels = len(channels_list)
        for i in range(num_levels):
            dilation_size = dilation_list[i]
            in_channels = input_dim if i == 0 else channels_list[i-1]
            out_channels = channels_list[i]
            # print("第"+str(i+1)+"层:",in_channels,out_channels)
            if i == num_levels-1:
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                         padding=(kernel_size - 1) * dilation_size, activation=activation, last_norm=True)]
            else:
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, activation=activation, last_norm=False)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # print("TCN_AE:",x.shape)
        return self.network(x)

class Mylinear(nn.Module):
    def __init__(self, in_channels, out_channels, activation, is_last = False):
        super(Mylinear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.activation = activation
        self.bn = nn.BatchNorm1d(out_channels)
        self.is_last = is_last
        

    def forward(self, x):
        # print(x.shape)
        # print(self.linear)
        # print(self.linear(torch.transpose(x,1,2)).shape)
        # print(x.shape)
        # print(x.dtype)
        # x = x[:,:50,:]
        # print(x.shape)
        # print(x.dtype)
        # if len(x.shape) == 2:
        #     device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        #     x = torch.stack([x.T]*4).to(device)
        #     x = x.to(self.linear.weight.dtype)
        
        # out = self.activation(self.linear(torch.transpose(x,1,2)))
        # if self.is_last:
        #     return out
        # # print(self.bn(torch.transpose(out,1,2)).shape)
        # return self.bn(torch.transpose(out,1,2))
    
        if len(x.shape) == 3:
            out = self.activation(self.linear(torch.transpose(x,1,2)))
            if self.is_last:
                return out
            return self.bn(torch.transpose(out,1,2))
        else:
            # print(x.shape)
            out = self.activation(self.linear(x))
            if self.is_last:
                return out
            return self.bn(out)
        

class MLPencoder(nn.Module):
    def __init__(self, layer_list, activation, config, last_norm = False):
        super(MLPencoder, self).__init__()
        self.activation = activation
        self.config = config
        layers = []
        layer_num = len(layer_list)
        self.device = config.device
        
        self.layer1 = Mylinear(layer_list[0],layer_list[1],activation)
        self.layer2 = Mylinear(layer_list[1], layer_list[2], activation)
        if last_norm:
            self.layer3 = Mylinear(layer_list[2], layer_list[3], activation)
        else:
            self.layer3 = Mylinear(layer_list[2], layer_list[3], activation,True)
        # self.layer4 = Mylinear(layer_list[3], layer_list[4], activation,True)
        # for i in range(1, layer_num):
        #     if i == layer_num - 1:
        #         layers += [Mylinear(layer_list[i-1],layer_list[i],activation,True)]
        #     else:
        #         layers += [Mylinear(layer_list[i-1],layer_list[i],activation)]
        # self.layers = layers
        # self.network = nn.Sequential(*layers)

    def forward(self, x):
        # res_ls = []
        # for i in range(len(self.layers)):
        #     if i == 0:
        #         res_ls.append(self.layers[i](x))
        #     else:
        #         res_ls.append(self.layers[i](res_ls[-1]))
        # print(x.type())
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not torch.is_tensor(x):
            x = torch.tensor(x).to(self.device)
        # print(x.shape)
        # print(x.dtype)
        # print(x[:2,:])
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        # x4 = self.layer4(x3)
        
        # return x3
        if len(x.shape) == 2:
            return x3#.cpu().detach().numpy()
        return x1, x2, x3#, x4


class Mymodel1(nn.Module):
    def __init__(self, config, embedding2y=True, device=None, *args, **kwargs):
        # 把所有的block组成的list作为整体传入，建立model(block之间的传输关系)
        super(Mymodel1,self).__init__()
        self.config = config  # 包含了模型初始化的一些参数设置
        self.embedding2y = embedding2y  # T or F
        if device is None:
            # device = 'cuda:2' if t.cuda.is_available() else 'cpu'
            device = config.device
        self.device = device

        # self.encoder_layers = None
        # self.decoder_layers = None

        dict_activation = {'relu': nn.ReLU(),
                           'softplus': nn.Softplus(),
                           'tanh': nn.Tanh(),
                           'selu': nn.SELU(),
                           'lrelu': nn.LeakyReLU(),
                           'prelu': nn.PReLU(),
                           'sigmoid': nn.Sigmoid()}

        self.activation = dict_activation[self.config.ACTIVATION]

        self.tc_loss = None
        self.embedding2y_layer = None

        # self.build_output_shape = None

        self.layers = self.build_layers()
        # self.config = config
        # self.embedding2y_layer = embedding2y_layer

    def _build_encoder_x_u(self):
        # return TCN_VAE_decoder(self.config.INPUT_DIM,self.config.ENCODER_NODES,
        #                        self.config.ENCODER_DILATION_RATES,activation=self.config.ACTIVATION)
        return MLPencoder([self.config.INPUT_DIM]+self.config.ENCODER_NODES, self.activation, self.config, True)

    def _build_encoder_x_z(self):
        # return TCN_VAE_decoder(self.config.INPUT_DIM,self.config.ENCODER_NODES,
        #                        self.config.ENCODER_DILATION_RATES,activation=self.config.ACTIVATION)
        return MLPencoder([self.config.INPUT_DIM] + self.config.ENCODER_NODES, self.activation, self.config, True)

    def _build_encoder_x_s(self):
        # return TCN_VAE_decoder(self.config.INPUT_DIM,self.config.ENCODER_NODES,
        #                        self.config.ENCODER_DILATION_RATES,activation=self.config.ACTIVATION)

        return MLPencoder([self.config.INPUT_DIM] + self.config.ENCODER_NODES, self.activation, self.config,True)

    def _build_u_encoder(self):
        return TCN_VAE_decoder(self.config.HIDDEN_V_DIM, self.config.encoder_nodes,
                               self.config.encoder_dilation_rates, activation=self.config.ACTIVATION)
    def _build_z(self):
        return TCN_VAE_decoder(self.config.HIDDEN_V_DIM, self.config.encoder_nodes,
                               self.config.encoder_dilation_rates, activation=self.config.ACTIVATION)
    def _build_s(self):
        return TCN_VAE_decoder(self.config.EMBEDDING_LEN, self.config.decoder_nodes,
                               self.config.decoder_dilation_rates, activation=self.config.ACTIVATION)
    def _build_u_decoder(self):
        return TCN_VAE_decoder(self.config.EMBEDDING_LEN, self.config.decoder_nodes,
                               self.config.decoder_dilation_rates, activation=self.config.ACTIVATION)

    def _build_decoder_x_u(self):
        # return TCN_VAE_decoder(self.config.HIDDEN_V_DIM,self.config.DECODER_NODES,
        #                        self.config.DECODER_DILATION_RATES,activation=self.config.ACTIVATION)

        return MLPencoder([self.config.HIDDEN_V_DIM] + self.config.DECODER_NODES, self.activation, self.config)

    def _build_decoder_x_z(self):
        # return TCN_VAE_decoder(self.config.HIDDEN_V_DIM,self.config.DECODER_NODES,
        #                        self.config.DECODER_DILATION_RATES,activation=self.config.ACTIVATION)
        return MLPencoder([self.config.HIDDEN_V_DIM] + self.config.DECODER_NODES, self.activation, self.config)

    def _build_decoder_x_s(self):
        # return TCN_VAE_decoder(self.config.HIDDEN_V_DIM,self.config.DECODER_NODES,
        #                        self.config.DECODER_DILATION_RATES,activation=self.config.ACTIVATION)
        return MLPencoder([self.config.HIDDEN_V_DIM] + self.config.DECODER_NODES, self.activation, self.config)



    def build_layers(self):
        print("Model build is called. ")
        # layer_list = nn.ModuleList()
        # self.encoder_layer_x_u = self._build_encoder_x_u()
        self.encoder_layer_x_z = self._build_encoder_x_z()
        self.encoder_layer_x_s = self._build_encoder_x_s()
        # self.encoder_u_layer = self._build_u_encoder()
        self.z_layer = self._build_z()

        # self.decoder_layer_x_u = self._build_decoder_x_u()
        self.decoder_layer_x_z = self._build_decoder_x_z()
        self.decoder_layer_x_s = self._build_decoder_x_s()
        # self.decoder_u_layer = self._build_u_decoder()
        self.s_layer = self._build_s()
        # layer_list.append(self.encoder_layer)
        # self.HIDDEN_V_DIM=16
        # layer_list.append(self.encoder_u_layer)
        # layer_list.append(self.encoder_u_layer)

        # 处理全连接层
        # layer_list.append(Multi_linear(self.config, self.activation, "encode"))

        if self.embedding2y:
            self.embedding2y_layer = Embedding2Y(self.config)

        # layer_list.append(Multi_linear(self.config, self.activation, "decode"))

        # layer_list.append(self.decoder_layer)
        # return layer_list

    # def __loss_fn(self):
    #     def RMSEloss(y, y_hat):
    #         mse = (y - y_hat) ** 2
    #         # mse = mask * mse
    #         mse = t.mean(mse)
    #         return mse ** 0.5

    def to_tensor(self, x: np.ndarray) -> t.Tensor:
        tensor = t.as_tensor(x, dtype=t.float32).to(self.device)
        return tensor

    def forward(self, x):
        # x = inputs # 90*50
        # print(x.dtype)
        # print(x.shape)
        # hid_u1, hid_u2, hid_u3 = self.encoder_layer_x_u(x)#[:-1]
        hid_z1, hid_z2, hid_z3 = self.encoder_layer_x_z(x)#[:-1]
        hid_s1, hid_s2, hid_s3 = self.encoder_layer_x_s(x)#[:-1]
        # hid_u3 = self.encoder_layer_x_u(x)#[:-1]
        # hid_z3 = self.encoder_layer_x_z(x)#[:-1]
        # hid_s3 = self.encoder_layer_x_s(x)#[:-1]
        # u = torch.transpose(self.encoder_layer_x_u(x)[-1],1,2) #16*50
        z = self.encoder_layer_x_z(x)[-1] # 32*50
        s = self.encoder_layer_x_s(x)[-1]
        # print(z.shape,s.shape)
        # u = torch.transpose(self.encoder_layer_x_u(x),1,2) #16*50
        # z = torch.transpose(self.encoder_layer_x_z(x),1,2)
        # s = torch.transpose(self.encoder_layer_x_s(x),1,2)
        # u = self.encoder_layer_x_u(x) #16*50
        # z = self.encoder_layer_x_z(x)
        # s = self.encoder_layer_x_s(x)
        # embedding_u = self.encoder_u_layer(u)
        # embedding_z = self.z_layer(z)
        # print(z.shape)
        embedding = self.z_layer(z)
        # embedding = (embedding_u+embedding_z)/2
        s_hat = self.s_layer(embedding)
        # u_hat = self.decoder_u_layer(embedding_u)
        rec_x_s = torch.transpose(self.decoder_layer_x_s(s)[-1],1,2)
        # rec_x_u = torch.transpose(self.decoder_layer_x_u(u_hat)[-1],1,2)
        rec_x_z = torch.transpose(self.decoder_layer_x_z(z)[-1],1,2)
        # rec_x_s = torch.transpose(self.decoder_layer_x_s(s),1,2)
        # rec_x_u = torch.transpose(self.decoder_layer_x_u(u_hat),1,2)
        # rec_x_z = torch.transpose(self.decoder_layer_x_z(z),1,2)
        # rec_x_s = self.decoder_layer_x_s(s_hat)
        # rec_x_u = self.decoder_layer_x_u(u_hat)
        # rec_x_z = self.decoder_layer_x_z(z)


        # return block_outputs
        if self.embedding2y:
            target_y = self.embedding2y_layer(torch.transpose(embedding,1,2))
            # target_y_z = self.embedding2y_layer(torch.transpose(embedding_z,1,2))
            # target_y_u = self.embedding2y_layer(torch.transpose(embedding_u,1,2))
            # print(target_y)
            return target_y, torch.transpose(embedding,1,2),rec_x_z,rec_x_s, z,s,s_hat,\
                   hid_z1, hid_z2, hid_z3,hid_s1, hid_s2, hid_s3
        else:
            return torch.transpose(embedding,1,2), rec_x_z,rec_x_s, z,s,s_hat,\
                   hid_z1, hid_z2, hid_z3,hid_s1, hid_s2, hid_s3


    # def predict_one_sample(self, input):
    #     # x.shape: 90*50
    #     x = torch.stack([input] * self.config.BATCH_SIZE)


