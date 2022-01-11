import cv2
import numpy as np

def draw_predect(image, trajectory, ped=0, color=(0, 0, 255), thickness=2): #画出预测轨迹点
    trajectory = trajectory.astype(np.int32)
    for each in trajectory:
        # print(each)
        cv2.circle(image, (each[0], each[1]), 2, color, thickness)
    return image


def draw_projected_box3d(image, qs, label=0, color=(0, 200, 0), thickness=2): #在图像中画三维框
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    qs = qs.astype(np.int32)
    text = "ID:"+str(label)
    cv2.putText(image, text, (min(qs[:, 0]), min(qs[:, 1]-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 2)

    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

    return image
