import sys
import os
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from process import mean_blur

def draw_scatter3d(xs, ys, zs):
    ax = plt.figure().add_subplot(111, projection = '3d')
    ax.scatter(xs, ys, zs, c='r', marker = 'o')
    # set axis label
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


if __name__ == "__main__":
    path1 = "/home/zavieton/3D/pedestrian_predict/tracking_predict/data_fold/data_withGauss"
    path2 = "/home/zavieton/3D/pedestrian_predict/tracking_predict/data_fold/data_truncation"
    for path in [path1]:
        filename = "0016_train_7338_0_169_with_gauss_1"
        data_path = os.path.join(path, filename)
        datas = []
        with open(data_path,'r') as f:
            for i, l in enumerate(f):
                fields = l.split(" ")
                datas.append([float(fields[0]), float(fields[1]), float(fields[2])])

        f.close()

        xs = []
        ys = []
        zs = []

        for each in datas:
            xs.append(each[0])
            ys.append(each[1])
            zs.append(each[2])

        xs = mean_blur(xs)
        ys = mean_blur(ys)
        zs = mean_blur(zs)

        draw_scatter3d(xs, ys, zs)

