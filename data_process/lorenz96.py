import numpy as np
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


#定义标准化函数
def z_score_normalize(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = (data - mean) / std_dev
    return normalized_data


def get_L96_func(N, F):
    def L96(t, x):
        """Lorenz 96 model with constant forcing"""
        # Setting up vector
        d = np.zeros(N)
        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        for i in range(N):
            d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
        return d
    return L96

# def L96(t, x):
#     """Lorenz 96 model with constant forcing"""
#     # Setting up vector
#     config = lorenz96_config.Lorenz96Config()
#     N = config.K
#     F = config.F
#     d = np.zeros(N)
#     # Loops over indices (with operations and Python underflow indexing handling edge cases)
#     for i in range(N):
#         d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
#     return d


def gen_L96_data(N, F, time_range=(0, 20), dt=0.02, skip_time_num = 1000,  x_init_way='norm', init_param={'mu': 0, 'sigma': 0.1}):
    """
    param x_init_way: str type, ones--> 全1初始化； norm --> 高斯分布初始化，
    param init_param: 初始化的参数
    """

    
    t_eval = np.arange(time_range[0], time_range[1], 0.001)  # 默认使用超小步长
    # print(len(t_eval))
    dt_factor = dt/0.001  # 设定的步长和默认步长的比值
    assert dt_factor == int(dt_factor), 'dt should be a multiple of 0.001'
    dt_factor = int(dt_factor)
    
    if x_init_way == 'ones':
        x0 = F * np.ones(N)  # Initial state (equilibrium)
        x0[0] += 0.01  # Add small perturbation to the first variable
    elif x_init_way == 'norm':
        np.random.seed(1)
        x0 = np.random.randn(N) * init_param['sigma'] + init_param['mu']
    elif x_init_way == 'uniform':
        np.random.seed(1)
        x0 = np.random.uniform(init_param['low'], init_param['high'], N)
    else:
        raise NotImplementedError()

    x = integrate.solve_ivp(get_L96_func(N, F), time_range, x0, t_eval=t_eval).y
    xx = x[:, ::dt_factor]
    # print(dt_factor)
    # print(x.shape)
    
    return xx[:, skip_time_num:].T  # 60, 100000

if __name__ == '__main__':
    N = 60  # Number of variables
    F = 5  # Forcing
    time_range = (0, 30)
    dt = 0.02

    data_x = gen_L96_data(N, F, time_range, dt)

    X = np.arange(*time_range, dt)
    Y = np.arange(N)
    print(data_x.shape)
    print(X.shape)
    print(Y.shape)
    XX, YY = np.meshgrid(Y, X)
    fig = plt.figure(figsize=(4, 6))
    plt.contourf(XX, YY, data_x.T, 100, cmap=mpl.colormaps['seismic'])
    plt.title(f'F={F}')
    plt.xlabel('X')
    plt.ylabel('t')
    plt.tight_layout()
    plt.show()