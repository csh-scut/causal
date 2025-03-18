import torch
class Myconfig(object):
    def __init__(self):
        ### 数据信息
        self.DATA_NAME = 'power_grid'
        # 输入维度（变量个数）
        self.INPUT_DIM = 120
        # 训练数据长度
        self.TRAIN_LEN = 30
        # 嵌入长度
        self.EMBEDDING_LEN = 10
        self.residx = 11
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.getDataSkipStep = self.TRAIN_LEN

        ### MLP
        self.HIDDEN_V_DIM = 32
        self.ENCODER_NODES = [128,64,self.HIDDEN_V_DIM]
        # self.ENCODER_DILATION_RATES = [1, 2, 4,8]
        self.DECODER_NODES = [64,128, self.INPUT_DIM]
        # self.DECODER_DILATION_RATES = [8,4, 2, 1]

        ### TCN
        self.encoder_nodes = [128, 64, 32, self.EMBEDDING_LEN]
        self.encoder_dilation_rates = [1, 2, 4,8]
        self.decoder_nodes = [32, 64, 128,self.HIDDEN_V_DIM]
        self.decoder_dilation_rates = [8,4, 2, 1]

        ### TCN
        # 每个TCN block的层数
        self.TCN_BLOCK_REPEAT_TIMES = 1
        self.KERNEL_SIZE = 3
        # 输入丢弃率，类似于resample
        self.INPUT_DROP_RATE = 0.0
        # 目标变量索引
        self.TARGET_IDX = 68
        self.DROPOUT = 0.1

        ### FC
        self.encoder_fc = [32,self.EMBEDDING_LEN]
        self.encoder_fc_layer_num = 2
        self.decoder_fc = [32,self.HIDDEN_V_DIM]
        self.decoder_fc_layer_num = 2

        ### model
        self.LOSS_WEIGHTS = {'masked_embedding_loss': 0.6,
                            #  'masked_embedding_loss': 0.6,

                             'future_consistency_loss': 0.15,
                            #  'future_consistency_loss': 0,

                             'reconstruction_loss_hidden':0.05,
                             'reconstruction_loss_x':0.05,
                            # 'reconstruction_loss_hidden':0,
                            #  'reconstruction_loss_x':0,

                             'orthogonal_loss':0.1,
                            #  'orthogonal_loss':0,
                             'tc_loss': 3.0}

        # 正则化方式
        self.KERNEL_REGULARIZER = 'l2'
        # self.WEIGHT_DECAY = {'l': 0.0}  # l1 / l2
        self.LR = 0.01
        self.LR_DECAY = 0.1
        # 批大小
        self.BATCH_SIZE = 8
        # 训练周期
        self.EPOCHES = 100
        # drop
        self.DROP_RATE = 0.0
        # 激活函数
        self.ACTIVATION = 'lrelu'
        # 层间归一化方法
        self.NORMALIZATION = 'ln'
        # 权重初始化方式
        self.KERNEL_INITIALIZER = 'he_normal'
        self.N_LR_DECAY_STEPS = 20
        self.WEIGHT_DECAY = 0
        self.EPOCH_SAMPLES = 1000

    # def change_target(target_idx):
    #     self.TARGET_IDX = target_idx