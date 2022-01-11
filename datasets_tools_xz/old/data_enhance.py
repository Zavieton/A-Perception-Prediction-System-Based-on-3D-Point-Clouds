import os
import sys
import numpy as np
import pandas as pd
import shutil
from process import mean_blur
import scipy.interpolate as spi
import matplotlib.pyplot as plt
from tqdm import tqdm
from colorama import Fore


KITTI_frequence = 0.1
rate = 5

def printlist(list):
    for each in list:
        print(each)


def interpolation(datas):
    time_seq = np.linspace(0, KITTI_frequence*len(datas), len(datas))
    lock = True
    datas = np.array(datas)
    x = datas[:, 0]
    y = datas[:, 1]
    z = datas[:, 2]

    def inter3(x, time_seq):
        ix3 = np.linspace(time_seq[0], time_seq[-1], rate*len(time_seq))
        tck = spi.splrep(time_seq, x, k=3)
        iy3 = spi.splev(ix3, tck)

        if not lock:
            plt.plot(time_seq, x)
            plt.plot(ix3, iy3)
            plt.show()

        return iy3

    ix = inter3(x, time_seq)
    iy = inter3(y, time_seq)
    iz = inter3(z, time_seq)

    ans = np.array([ix, iy, iz])
    ans = ans.T
    return ans


def data_increase(data_dir):
    with open(data_dir, "r") as f:
        datas = []
        for i, l in enumerate(f):
            fields = l.split(" ")
            datas.append([float(fields[3]), float(fields[4]), float(fields[5])])
    f.close()
    datas = np.array(datas)

    xs = datas[:, 0]
    ys = datas[:, 1]
    zs = datas[:, 2]

    xs = mean_blur(xs)
    ys = mean_blur(ys)
    zs = mean_blur(zs)

    datap = []

    for i in range(len(xs)):
        datap.append([xs[i], ys[i], zs[i]])

    datap.pop()

    # velocity
    vx0 = (datap[1][0] - datap[0][0])/KITTI_frequence
    vy0 = (datap[1][1] - datap[0][1])/KITTI_frequence
    vz0 = (datap[1][2] - datap[0][2])/KITTI_frequence

    datap[0] += [vx0, vy0, vz0]
    for i in range(1, len(datap)):
        vx = (datap[i][0] - datap[i-1][0])/KITTI_frequence
        vy = (datap[i][1] - datap[i-1][1])/KITTI_frequence
        vz = (datap[i][2] - datap[i-1][2])/KITTI_frequence
        datap[i] += [vx, vy, vz]

    # acceleration
    ax0 = (datap[1][3] - datap[0][3])/KITTI_frequence
    ay0 = (datap[1][4] - datap[0][4])/KITTI_frequence
    az0 = (datap[1][5] - datap[0][5])/KITTI_frequence

    datap[0] += [ax0, ay0, az0]
    for i in range(1, len(datap)):
        ax = (datap[i][3] - datap[i-1][3])/KITTI_frequence
        ay = (datap[i][4] - datap[i-1][4])/KITTI_frequence
        az = (datap[i][5] - datap[i-1][5])/KITTI_frequence
        datap[i] += [ax, ay, az]

    Data = interpolation(datap)

    xs = Data[:, 0]
    ys = Data[:, 1]
    zs = Data[:, 2]

    xs = mean_blur(xs, kernel_size=9)
    ys = mean_blur(ys, kernel_size=9)
    zs = mean_blur(zs, kernel_size=9)

    Data = []

    for i in range(len(xs)):
        Data.append([xs[i], ys[i], zs[i]])

    return Data


if __name__ == "__main__":
    data_save_dir = "/home/zavieton/3D/pedestrian_predict/tracking_predict/data_fold/data_enhance"
    data_dir = "/home/zavieton/3D/pedestrian_predict/tracking_predict/data_fold/data_raw"

    if os.path.exists(data_save_dir):
        shutil.rmtree(data_save_dir)
        os.mkdir(data_save_dir)
    else:
        os.mkdir(data_save_dir)


    if not os.path.exists(data_dir):
        raise Exception('Program ERR: Dir is not exist')
    else:
        print("Path is {}".format(data_dir))
        g = os.walk(data_dir)

        for path, _, file_list in g:
            lens = len(file_list)
            print("sample number is {}".format(lens))
            for i in tqdm(range(lens), bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.WHITE, Fore.WHITE)):
                cur_path = os.path.join(path, file_list[i])
                Data = data_increase(cur_path)
                save_path = os.path.join(data_save_dir, file_list[i])
                for s in Data:
                    with open(save_path, "a+") as fi:
                        strs = str(s[0]) + " " + str(s[1]) + " " + str(s[2])
                        fi.write('%s\n' % (strs))
                    fi.close()

        print("**********finished!***********")