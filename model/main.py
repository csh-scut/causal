from model.unit import Embedding2Y
import torch
import torch as t
import torch.nn as nn
from model.unit import Mymodel1,TCN_VAE_decoder,TCN_VAE_encoder,Mylinear,Multi_TCN_VAE,Multi_linear
from model.loss import RMSELoss, MaskEmbeddingLoss
from torch import optim
import numpy as np
import time


class Mymodel(object):
    def __init__(self, config, embedding2y=True, device=None, *args, **kwargs):
        super(Mymodel, self).__init__(*args, **kwargs)

        self.config = config # 包含了模型初始化的一些参数设置
        self.embedding2y = embedding2y # T or F
        if device is None:
            device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.device = device

        self.encoder_layers = None  
        self.decoder_layers = None  

        dict_activation = { 'relu': nn.ReLU(),
                            'softplus': nn.Softplus(),
                            'tanh': nn.Tanh(),
                            'selu': nn.SELU(),
                            'lrelu': nn.LeakyReLU(),
                            'prelu': nn.PReLU(),
                            'sigmoid': nn.Sigmoid()}
        
        self.activation = dict_activation[self.config.ACTIVATION]

        self.tc_loss = None 
        self.embedding2y_layer = None

        self.build_output_shape = None

    def _build_encoder(self):
        return Multi_TCN_VAE(2,self.config,"encode")

                    
    def _build_decoder(self):
        # if self.config.DECODER_NODES is not None and len(self.config.DECODER_NODES) > 0:
        #     decoder_nodes = self.config.DECODER_NODES
        # else:
        #     decoder_nodes = self.config.ENCODER_NODES[:-1][::-1]
        #     decoder_nodes.append(self.config.INPUT_DIM)

        return Multi_TCN_VAE(2,self.config,"decode")
    
    def build_layers(self):
        print("Model build is called. ")
        layer_list = nn.ModuleList()
        encoder_layer = self._build_encoder()
        decoder_layer = self._build_decoder()
        layer_list.append(encoder_layer)
        # self.HIDDEN_V_DIM=16
        
        # 处理全连接层
        layer_list.append(Multi_linear(self.config,self.activation,"encode",num=2))

        if self.embedding2y:
            self.embedding2y_layer = Embedding2Y(self.config)

        layer_list.append(Multi_linear(self.config,self.activation,"decode",num=1))

        layer_list.append(decoder_layer)
        return layer_list

    # def __loss_fn(self):
    #     def RMSEloss(y, y_hat):
    #         mse = (y - y_hat) ** 2
    #         # mse = mask * mse
    #         mse = t.mean(mse)
    #         return mse ** 0.5


    def to_tensor(self, x: np.ndarray) -> t.Tensor:
        tensor = t.as_tensor(x, dtype=t.float32).to(self.device)
        return tensor

    def fit(self, inputs):
        
        self.block_list = self.build_layers()
        print(self.block_list)
        self.model = Mymodel1(self.block_list,self.config,self.embedding2y_layer).to(self.device)
        n_epoches = self.config.EPOCHES

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.LR, weight_decay=self.config.WEIGHT_DECAY)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.config.N_LR_DECAY_STEPS, gamma=self.config.LR_DECAY)
        # training_loss_fn = self.__loss_fn(self.loss)

        print('\n')
        print('='*30+' Start fitting '+'='*30)

        start = time.time()

        for key in self.config.LOSS_WEIGHTS:
            if key == "masked_embedding_loss":
                w_mask_embedding = self.config.LOSS_WEIGHTS[key]
            elif key == 'prediction_loss':
                w_pred = self.config.LOSS_WEIGHTS[key]
            elif key == 'reconstruction_loss_hidden':
                w_resconstruct_u = self.config.LOSS_WEIGHTS[key]
            elif key == 'reconstruction_loss_x':
                w_resconstruct_x = self.config.LOSS_WEIGHTS[key]
            elif key == 'consistency_loss':
                w_consistency = self.config.LOSS_WEIGHTS[key]
            else:
                continue
        epoch = 0
        epoches_loss_ls = []
        train_loss_ls = []
        prediction_loss_ls = []
        embedding_loss_ls = []
        prediction_loss_z_ls = []
        prediction_loss_u_ls = []
        embedding_loss_z_ls = []
        embedding_loss_u_ls = []
        consistency_loss_ls = []
        reconstruction_loss_x_ls = []
        reconstruction_loss_u_ls = []


        while (epoch < n_epoches):
            
            epoch +=1

            b_loss_ls = []
            b_pred_loss_ls = []
            b_embed_loss_ls = []
            b_pred_loss_z_ls = []
            b_pred_loss_u_ls = []
            b_embed_loss_z_ls = []
            b_embed_loss_u_ls = []
            b_consistency_loss_ls = []
            b_rec_loss_x_ls = []
            b_rec_loss_u_ls = []
            for batch in iter(inputs):
                # iteration += 1
                # if (iteration > n_iterations) or (break_flag):
                #     continue

                self.model.train()
                # Parse batch
                batched_input_data       = batch['batched_input_data']
                batched_target_embedding = batch['batched_target_embedding']
                batched_target_y         = batch['batched_target_y']
                # print(batched_input_data.requires_grad_)

                optimizer.zero_grad()
                # optimizer.step()
                # print()

                predict_y, predict_y_z, predict_y_u, block_outputs   = self.model(batched_input_data)
                # print(batched_target_y[0],predict_y[0])
                # print("---------------------")
                # print(batched_target_y[-1],predict_y[-1])
                # print()
                # print(block_outputs[0][0,0,:],torch.transpose(block_outputs[2],1,2)[0,0,:])
                # print(len(block_outputs))
                # print(block_outputs[0].shape," ---") 4*64*50
                # print(block_outputs[1].shape," ---") 4*50*16
                # print(block_outputs[2].shape," ---") 4*50*64
                # print(block_outputs[3].shape) 4*90*50
                # print('+++++++++++++++++++')


                # inputs: 4*90*50
                # print(RMSELoss(batched_target_y, predict_y),RMSELoss(batched_input_data, block_outputs[-1]),RMSELoss(block_outputs[0],torch.transpose(block_outputs[2],1,2)))

                predict_loss_z = RMSELoss(batched_target_y, predict_y_z)
                predict_loss_u = RMSELoss(batched_target_y, predict_y_u)
                predict_loss = RMSELoss(batched_target_y, predict_y)
                # print(batched_target_embedding.shape)
                embedding_loss_z = MaskEmbeddingLoss(batched_target_embedding, block_outputs[1][0])
                embedding_loss_u = MaskEmbeddingLoss(batched_target_embedding, block_outputs[1][1])
                embedding = (block_outputs[1][0] + block_outputs[1][1])/2
                embedding_loss = MaskEmbeddingLoss(batched_target_embedding, embedding)
                consistency_loss = RMSELoss(block_outputs[1][0],block_outputs[1][1])
                reconstruct_loss_u = RMSELoss(block_outputs[0][1],block_outputs[2][1])
                reconstruct_loss_x = RMSELoss(batched_input_data, block_outputs[-1][0]) + RMSELoss(batched_input_data, block_outputs[-1][1])

                b_pred_loss_ls.append(predict_loss)
                b_embed_loss_ls.append(embedding_loss)
                b_pred_loss_z_ls.append(predict_loss_z)
                b_pred_loss_u_ls.append(predict_loss_u)
                b_embed_loss_z_ls.append(embedding_loss_z)
                b_embed_loss_u_ls.append(embedding_loss_u)
                b_consistency_loss_ls.append(consistency_loss)
                b_rec_loss_x_ls.append(reconstruct_loss_x)
                b_rec_loss_u_ls.append(reconstruct_loss_u)

                batch_train_loss = w_mask_embedding * (embedding_loss + embedding_loss_u + embedding_loss_z) + \
                    w_pred * (predict_loss + predict_loss_u + predict_loss_z) + w_resconstruct_u * reconstruct_loss_u +\
                    w_resconstruct_x * reconstruct_loss_x + w_consistency * consistency_loss
                # print(batch_train_loss)
                b_loss_ls.append(batch_train_loss)


                # Protection if exploding gradients
                # if not np.isnan(float(training_loss)):
                batch_train_loss.backward()
                t.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            # print(optimizer.param_groups)
            lr_scheduler.step()

            print("学习率",optimizer.param_groups[0]['lr'])


            # 一次 epoch 结束
            print("第" + str(epoch) + "个epoch:", end=" ")
            print(sum(b_loss_ls)/len(b_loss_ls))
            # print("++++++++++++++++++++++++++")
            epoches_loss_ls.append(sum(b_loss_ls)/len(b_loss_ls))
            prediction_loss_ls.append(sum(b_pred_loss_ls)/len(b_pred_loss_ls))
            embedding_loss_ls.append(sum(b_embed_loss_ls)/len(b_embed_loss_ls))
            embedding_loss_z_ls.append(sum(b_embed_loss_z_ls)/len(b_embed_loss_z_ls))
            embedding_loss_u_ls.append(sum(b_embed_loss_u_ls)/len(b_embed_loss_u_ls))
            prediction_loss_z_ls.append(sum(b_pred_loss_z_ls)/len(b_pred_loss_z_ls))
            prediction_loss_u_ls.append(sum(b_pred_loss_u_ls)/len(b_pred_loss_u_ls))
            consistency_loss_ls.append(sum(b_consistency_loss_ls)/len(b_consistency_loss_ls))
            reconstruction_loss_x_ls.append(sum(b_rec_loss_x_ls)/len(b_rec_loss_x_ls))
            reconstruction_loss_u_ls.append(sum(b_rec_loss_u_ls)/len(b_rec_loss_u_ls))
            # embedding_loss_ls.append(sum(b_embed_loss)/len(b_embed_loss))
            # embedding_loss_ls.append(sum(b_embed_loss)/len(b_embed_loss))
            # embedding_loss_ls.append(sum(b_embed_loss)/len(b_embed_loss))
            # embedding_loss_ls.append(sum(b_embed_loss)/len(b_embed_loss))

            train_loss_ls.append(b_loss_ls)
            if inputs.shuffle:
                np.random.shuffle(inputs.select_idxs)
        #End of fitting
        # if n_iterations >0:
        #     # This is batch loss!
        #     self.final_insample_loss = np.float(training_loss.cpu().data.numpy()) if not break_flag else best_insample_loss
        #     string = 'Step: {}, Time: {:03.3f}, Insample {}: {:.5f}'.format(iteration,
        #                                                                     time.time()-start,
        #                                                                     self.loss,
        #                                                                     self.final_insample_loss)
        #     if val_ts_loader is not None:
        #         self.final_outsample_loss = self.evaluate_performance(ts_loader=val_ts_loader,
        #                                                               validation_loss_fn=validation_loss_fn)
        #         string += ", Outsample {}: {:.5f}".format(self.val_loss, self.final_outsample_loss)
            # print(string)
            # print('='*30+'  End fitting  '+'='*30)
            # print('\n')
        self.train_loss = train_loss_ls
        self.epoches_loss = epoches_loss_ls
        self.pred_loss = prediction_loss_ls
        self.embed_loss = embedding_loss_ls
        self.pred_loss_z = prediction_loss_z_ls
        self.pred_loss_u = prediction_loss_u_ls
        self.embed_loss_z = embedding_loss_z_ls
        self.embed_loss_u = embedding_loss_u_ls
        self.consistency_loss = consistency_loss_ls
        self.rec_loss_x = reconstruction_loss_x_ls
        self.rec_loss_u = reconstruction_loss_u_ls

        string = 'Time: {:03.3f}'.format(time.time()-start)
        print(string)
        print('='*30+'  End fitting  '+'='*30)
        print('\n')


    def predict(self, inputs):
        # inputs: m个时间点的数据，预测未来m+L-1的值
        self.model.eval()
        # assert not ts_loader.shuffle, 'ts_loader must have shuffle as False.'

        # x = torch.unsqueeze(inputs, dim=0)
        predict_y, predict_y_z, predict_y_u, block_outputs = self.model(inputs, self.block_list)

        # self.model.train()
        return predict_y, predict_y_z, predict_y_u ,block_outputs
        # if return_decomposition:
        #     return outsample_ys, forecasts, block_forecasts, outsample_masks
        # else:
        #     return outsample_ys, forecasts, outsample_masks


        