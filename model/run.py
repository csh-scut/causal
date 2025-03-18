# import os
import pandas as pd
if __name__ == '__main__':
    dir = "../RS_results/lorenz96/y2(old_setting).xlsx"
    n = len(dir)
    # print(n)
    # if n == 49:
    #     target = int(dir[-19])
    # elif n == 50:
    #     target = int(dir[-20]+dir[-19])
    # else:
    #     target = int(dir[-21]+dir[-20]+dir[-19])

    if n == 43:
        target = int(dir[-19])
    else:
        target = int(dir[-20]+dir[-19])
    # print(n)
    print(target)

    # net_dir = "/home/csh/CReP/self_supervised_two_parts/data_files/power_grid/E_cause_36to33.csv"
    # net = pd.read_csv(net_dir,index_col=0)
    # cause = []
    # effect = []
    # for edge in net.values:
    #     if edge[0] == target-1:
    #         effect.append(edge[1]+1)
    #     if edge[1] == target-1:
    #         cause.append(edge[0]+1)

    # print("cause:",cause)
    # cause_var = [target]
    # if 
    # target = dir
    rc_power = pd.read_excel(dir,index_col=0).drop("G"+str(target))
    # print(rc_power.index.values)
    # print("cause strength:")
    print(rc_power.sort_values("S_RC",ascending=False).head(10))
    # print("effect strength:")
    # print("effect:",effect)
    print(rc_power.sort_values("Z_RC",ascending=False).head(10))
    # os.system("python train2.py")

    # print("end train2.py")
    # os.system("python all_LRP.py")
    # print("end all_LRP.py")