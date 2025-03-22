# CReP: Causal-oriented representation learning for time-series forecasting based on the spatiotemporal information transformation
This respository includes codes and datasets for paper "Causal-oriented representation learning for time-series forecasting based on the spatiotemporal information transformation".

A causal-oriented representation learning predictor was developed to accurately make multi-step-ahead predictions of a time series and effectively infer the causal dependencies of a target variable by combining dynamic causation and spatiotemporal information (STI) transformation.

## Data  availability
- The real world datasets can be downloaded here.
- The Lorenz 96 dataset can be generated using the function `get_L96_func` in `data_process/lorenz96.py`.
- The raw DREAM4 dataset can be generated from `InSilicoSize50-Yeast1` of DREAM_Challenges in the software `GeneNetWeaver`. The parameter settings are provided in Supplementary Section 6.
- The power grid dataset can be generated using the code from the paper "Higher-order Granger reservoir computing: simultaneously achieving scalable complex structures inference and accurate dynamics prediction". The net for the system used to generate the dataset is in the `data_files/power_grid`. The parameter settings are provided in Supplementary Section 6.


## Environment requirements
- torch==2.5.1
- numpy==1.26.4
- networkx==3.4.2
- panda==0.3.1
- scipy==1.15.1

## Training: making predictions and interpreting causal representations
- we release the sample training codes and predicting codes corresponding to the Lorenz 96 dataset, which should be generated 
 and saved at `data_files/lorenz96/` in advance. The script `train2.py` is used for training and the script `all_LRP.py` is used for interpreting the learned causal representations of trained model by `train2.py`.
- For simulated datasets, the script `test_all_data.py` is used for evaluation after training the model.

