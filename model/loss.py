import torch.nn as nn
import model.unit as unit
import torch
import torch as t
import numpy as np
from model.unit import embedding_2_predict_y
from scipy.stats import pearsonr



def Y_2_embedding(y,sample_time_len,embedding_len):
    '''input:torch.tensor([,,,]),length:m'''
    if len(y)!=sample_time_len:
        print("the training length of y is not equal to sample_time_len!")
        return
    embedding_y = torch.tensor(np.zeros((sample_time_len,embedding_len)))
    embedding_y[0,0] = y[0]
    # same_idxs = []
    for start_row_idx in range(1, sample_time_len):
        # idxs = []
        col_num = min(start_row_idx + 1, embedding_len)
        for col_idx in range(col_num):
            idx = np.zeros((1, 2), dtype=np.int32)
            idx[0, 1] = col_idx
            idx[0, 0] = start_row_idx - col_idx
            embedding_y[idx[0,0],idx[0,1]] = y[start_row_idx]
            # idxs.append(idx)
        # same_idxs.append(np.concatenate(idxs))
    return embedding_y



def get_known_mask(time_len, embedding_len):
    """
    针对的矩阵是[sample_time_len, embedding_len]
    已知的赋值为1，未知的为0
    :param time_len:
    :param embedding_len:
    :return:
    """
    mask = np.ones(shape=[time_len, embedding_len], dtype=np.float32)

    # TODO：gene数据应该需要这个
    # mask = (mask * np.arange(embedding_len)[::-1]).astype(dtype=np.float32)

    for i in range(embedding_len - 1):
        mask[time_len - embedding_len + 1 + i, -(i+1):] = 0.0

    return mask

def MaskEmbeddingLoss(true_y_embedding, embedding_Y, device):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shape1 = true_y_embedding.shape
    shape2 = embedding_Y.shape
    batch_size = true_y_embedding.shape[0]
    if shape1 != shape2:
        print("embedding size is not the same!")
        print(shape1,shape2)
        return
    train_len = true_y_embedding.shape[-2]
    embedding_len = true_y_embedding.shape[-1]
    mask_known_embedding = torch.tensor(get_known_mask(train_len,embedding_len)).to(device)
    loss = ((true_y_embedding-embedding_Y) * mask_known_embedding) ** 2
    # loss = loss * mask_known_embedding
    loss = t.sum(loss) / (t.sum(mask_known_embedding)*batch_size)
    return loss ** 0.5

def FutureConsistencyLoss(embedding_Y,config):
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    device = config.device
    batch_size, train_len, embedding_len = embedding_Y.shape
    mask_known_embedding = 1-torch.tensor(get_known_mask(train_len, embedding_len)).to(device)
    batched_target_embedding = []
    mean_y = embedding_2_predict_y(embedding_Y,config,all_y=False).to(device)
    # print(mean_y.shape)
    for k in range(batch_size):
        total_y = torch.cat([torch.tensor([0]*train_len).to(device),mean_y[k]])
        target_embedding = []
        for i in range(train_len):  # 50
            target_embedding.append(total_y[i:i + embedding_len])
        target_embedding = torch.stack(target_embedding)  # 默认竖着堆砌
        batched_target_embedding.append(target_embedding)

    batched_target_embedding = torch.stack(batched_target_embedding).to(device)
    loss = ((batched_target_embedding - embedding_Y)*mask_known_embedding) ** 2
    # loss = loss * mask_known_embedding
    loss = t.sum(loss) / (t.sum(mask_known_embedding)*batch_size)
    return loss**0.5

def RMSELoss(y, y_hat):
    """MSE Loss

    Calculates Mean Squared Error between
    y and y_hat. MAPE measures the relative prediction
    accuracy of a forecasting method by calculating the
    percentual deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series.

    Parameters
    ----------
    y: tensor (batch_size, output_size)
        actual values in torch tensor.
    y_hat: tensor (batch_size, output_size)
        predicted values in torch tensor.
    mask: tensor (batch_size, output_size)
        specifies date stamps per serie
        to consider in loss

    Returns
    -------
    mse:
    Mean Squared Error.
    """
    # y = torch.nn.functional.normalize(y,p=1.0,dim=-1)
    # y_hat = torch.nn.functional.normalize(y_hat,p=1.0,dim=-1)
    mse = (y - y_hat)**2
    # mse = mask * mse
    mse = t.mean(mse)
    return mse**0.5
    # def compute_output_shape(self, input_shape):
    #     return [input_shape[0], self.config.EMBEDDING_LEN - 1]

def PCCLoss(y1, y2):
    batch_size = y1.shape[0]
    pcc_ls =  []
    for k in range(batch_size):
        pcc_ls.append(np.corrcoef([y1[k].cpu().detach().numpy(),y2[k].cpu().detach().numpy()])[0,1])
    return sum(pcc_ls)/len(pcc_ls)

def OrthogonalLoss(x1,x2):
    # x1.shape:32*50
    batch_size = x1.shape[0]
    vector_num = x1.shape[1]
    res = 0
    for k in range(batch_size):
        innerProductMatrix = x1[k] @ x2[k].T
        res += ((innerProductMatrix ** 2).sum() / vector_num) ** 0.5
    return res / batch_size

# def RMSELoss_inner(y, y_hat):

# class ZeroLoss(tf.keras.losses.Loss):
#     def call(self, y_true, y_pred):
#         return 0.0


# class MaskEmbeddingLoss(tf.keras.losses.Loss):
#     def __init__(self, config):
#         """

#         :param config:
#         """
#         self.known_mask = tf.convert_to_tensor(get_known_mask(config.TRAIN_LEN, config.EMBEDDING_LEN))
#         self.config = config
#         super(MaskEmbeddingLoss, self).__init__()

#     def call(self, y_true, y_pred):
#         y_true = tf.convert_to_tensor(y_true)
#         y_pred = tf.cast(y_pred, y_true.dtype)
#         mask_embedding_loss = tf.reduce_mean(tf.multiply(
#             tf.square(y_true - y_pred),
#             self.known_mask)
#         ) * self.config.LOSS_WEIGHTS.get('masked_embedding_loss', 1.0)
#         return mask_embedding_loss


# class ReconstructionLoss(tf.keras.losses.Loss):
#     def __init__(self, config):
#         self.config = config
#         super(ReconstructionLoss, self).__init__()

#     def call(self, y_true, y_pred):
#         y_true = tf.convert_to_tensor(y_true)
#         y_pred = tf.cast(y_pred, y_true.dtype)

#         reconstruction_loss = tf.reduce_mean(
#             tf.square(y_true - y_pred)) * self.config.LOSS_WEIGHTS.get('reconstruction_loss', 1.0)

#         return reconstruction_loss


# def get_known_mask(time_len, embedding_len):
#     """
#     针对的矩阵是[sample_time_len, embedding_len]
#     已知的赋值为1，未知的为0
#     :param time_len:
#     :param embedding_len:
#     :return:
#     """
#     mask = np.ones(shape=[time_len, embedding_len], dtype=np.float32)

#     # TODO：gene数据应该需要这个
#     # mask = (mask * np.arange(embedding_len)[::-1]).astype(dtype=np.float32)

#     for i in range(embedding_len - 1):
#         mask[time_len - embedding_len + 1 + i, -(i+1):] = 0.0

#     return mask

# class TimeConsistencyLoss(tf.keras.layers.Layer):
#     def __init__(self, loss_weight, train_len, embedding_len, batch_size, **kwargs):
#         self.loss_weight = loss_weight
#         self.train_len = train_len
#         self.embedding_len = embedding_len
#         self.same_idxs = utils.get_same_idxs(train_len, embedding_len)
#         self.batch_size = batch_size
#         super(TimeConsistencyLoss, self).__init__(**kwargs)

#     def call(self, inputs, **kwargs):

#         batch_consistent_losses = []
#         for b in range(self.batch_size):
#             embedding = inputs[b]

#             consistent_loss = []
#             # 计算对角线元素相等
#             for i, same_idx in enumerate(self.same_idxs):
#                 same_y_s = tf.gather_nd(embedding, same_idx)
#                 mean_y = tf.reduce_mean(same_y_s)
#                 loss = tf.reduce_mean(tf.square(same_y_s - mean_y))
#                 consistent_loss.append(loss)

#             consistent_loss = tf.stack(consistent_loss)
#             consistent_loss = tf.reduce_mean(consistent_loss)
#             batch_consistent_losses.append(consistent_loss)

#         batch_consistent_losses = tf.stack(batch_consistent_losses)
#         tc_loss = tf.reduce_mean(batch_consistent_losses)

#         self.add_loss(tc_loss * self.loss_weight)
#         self.add_metric(tc_loss, name='time_consistency_loss', aggregation='mean')
#         return inputs


# class Y_Predict_RMSE_Metric(tf.keras.metrics.Metric):
#     def __init__(self, name='predict_y_rmse', **kwargs):
#         super(Y_Predict_RMSE_Metric, self).__init__(name=name, **kwargs)

#         self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())
#         self.rmse = self.add_weight(name='rmse', initializer=tf.zeros_initializer())

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = tf.convert_to_tensor(y_true)
#         y_pred = tf.cast(y_pred, y_true.dtype)
#         rmse = tf.reduce_sum(tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)))

#         self.count.assign_add(tf.shape(y_true)[0])
#         self.rmse.assign_add(rmse)

#     def result(self):
#         return self.rmse / tf.cast(self.count, tf.float32)

#     def reset_states(self):
#         self.count.assign(0)
#         self.rmse.assign(0)


# class MaskEmbeddingLossMetric(tf.keras.metrics.Metric):
#     def __init__(self, c, name='mask_embedding_loss', **kwargs):
#         super(MaskEmbeddingLossMetric, self).__init__(name=name, **kwargs)
#         self.known_mask = tf.convert_to_tensor(get_known_mask(c.TRAIN_LEN, c.EMBEDDING_LEN))

#         self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())
#         self.me_loss = self.add_weight(name='me_loss', initializer=tf.zeros_initializer())
#         self.loss_weight = c.LOSS_WEIGHTS.get('masked_embedding_loss', 1.0)

#         self.c = c

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = tf.convert_to_tensor(y_true)
#         y_pred = tf.cast(y_pred, y_true.dtype)
#         mask_embedding_loss = tf.reduce_sum(
#             tf.reduce_mean(tf.multiply(tf.square(y_true - y_pred),
#                                        self.known_mask), axis=[-1, -2])) * self.loss_weight

#         self.count.assign_add(tf.shape(y_true)[0])
#         self.me_loss.assign_add(mask_embedding_loss)

#     def result(self):
#         return self.me_loss / tf.cast(self.count, tf.float32)

#     def reset_states(self):
#         self.count.assign(0)
#         self.me_loss.assign(0)

#     def get_config(self):
#         config = super(MaskEmbeddingLossMetric, self).get_config()
#         config['c'] = self.c
#         return config


# class ReconstructionLossMetric(tf.keras.metrics.Metric):
#     def __init__(self, c, name='reconstruction_loss', **kwargs):
#         super(ReconstructionLossMetric, self).__init__(name=name, **kwargs)

#         self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())
#         self.r_loss = self.add_weight(name='r_loss', initializer=tf.zeros_initializer())
#         self.loss_weight = c.LOSS_WEIGHTS.get('reconstruction_loss', 1.0)

#         self.c = c

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = tf.convert_to_tensor(y_true)
#         y_pred = tf.cast(y_pred, y_true.dtype)
#         reconstruction_loss = tf.reduce_sum(
#             tf.reduce_mean(tf.square(y_true - y_pred), axis=[-1, -2])) * self.loss_weight

#         self.count.assign_add(tf.shape(y_true)[0])
#         self.r_loss.assign_add(reconstruction_loss)

#     def result(self):
#         return self.r_loss / tf.cast(self.count, tf.float32)

#     def reset_states(self):
#         self.count.assign(0)
#         self.r_loss.assign(0)

#     def get_config(self):
#         config = super(ReconstructionLossMetric, self).get_config()
#         config['c'] = self.c
#         return config