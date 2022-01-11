from kitti_calib import kitti_calib_lines
import numpy as np


def point_pcl_to_image(point): #将三维坐标点映射到图像坐标系
    velo_box = np.vstack(point) #(n,3)
    velo_box = velo_box[:, [0, 2, 1]]
    velo_box = np.concatenate((velo_box, np.ones(velo_box.shape[0]).reshape(-1, 1)), axis=1) #最后一列加入齐次项

    for calib_line in kitti_calib_lines:
        if 'P2' in calib_line:
            P2 = calib_line.split(' ')[1:]
            P2 = np.array(P2, dtype='float').reshape(3, 4)
        elif 'R0_rect' in calib_line:
            R0_rect = np.zeros((4, 4))
            R0 = calib_line.split(' ')[1:]
            R0 = np.array(R0, dtype='float').reshape(3, 3)
            R0_rect[:3, :3] = R0
            R0_rect[-1, -1] = 1
        elif 'velo_to_cam' in calib_line:
            velo_to_cam = np.zeros((4, 4))
            velo2cam = calib_line.split(' ')[1:]
            velo2cam = np.array(velo2cam, dtype='float').reshape(3, 4)
            velo_to_cam[:3, :] = velo2cam
            velo_to_cam[-1, -1] = 1

    tran_mat = P2.dot(R0_rect).dot(velo_to_cam)  # 3x4

    velo_box = velo_box.reshape(-1, 4).T  # 4xn
    img_box = np.dot(tran_mat, velo_box).T  # nx3

    # img_box[:, 2] = - np.abs(img_box[:,  2])
    # img_box = np.abs(img_box)

    img_box[:, 1] = img_box[:, 1] / img_box[:, 2]
    img_box[:, 0] = img_box[:, 0] / img_box[:, 2]
    # print(img_box)

    img_box = img_box[:, :2]  # （n,2）
    img_box = np.squeeze(img_box[:, :2])  # （n,2）
    return img_box


def box_calculator(pred_dicts): #计算三维坐标框到二维图像坐标系的映射
    pred_boxes = pred_dicts[:, 2:8] #x y z h w l heading

    #pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
    #pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
    #pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()

    """
            7 -------- 4
           /|         /|
          6 -------- 5 .
          | |        | |
          . 3 -------- 0
          |/         |/
          2 -------- 1
        Args:
            boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    """

    velo_box = []

    for i in range(pred_boxes.shape[0]):
        point1 = [pred_boxes[i][0] + 0.5 * pred_boxes[i][3], pred_boxes[i][1] + 0.5 * pred_boxes[i][4],
                  pred_boxes[i][2] + 0.5 * pred_boxes[i][5], 1]

        point2 = [pred_boxes[i][0] - 0.5 * pred_boxes[i][3], pred_boxes[i][1] + 0.5 * pred_boxes[i][4],
                  pred_boxes[i][2] + 0.5 * pred_boxes[i][5], 1]

        point3 = [pred_boxes[i][0] - 0.5 * pred_boxes[i][3], pred_boxes[i][1] - 0.5 * pred_boxes[i][4],
                  pred_boxes[i][2] + 0.5 * pred_boxes[i][5], 1]

        point4 = [pred_boxes[i][0] + 0.5 * pred_boxes[i][3], pred_boxes[i][1] - 0.5 * pred_boxes[i][4],
                  pred_boxes[i][2] + 0.5 * pred_boxes[i][5], 1]

        point5 = [pred_boxes[i][0] + 0.5 * pred_boxes[i][3], pred_boxes[i][1] + 0.5 * pred_boxes[i][4],
                  pred_boxes[i][2] - 0.5 * pred_boxes[i][5], 1]

        point6 = [pred_boxes[i][0] - 0.5 * pred_boxes[i][3], pred_boxes[i][1] + 0.5 * pred_boxes[i][4],
                  pred_boxes[i][2] - 0.5 * pred_boxes[i][5], 1]

        point7 = [pred_boxes[i][0] - 0.5 * pred_boxes[i][3], pred_boxes[i][1] - 0.5 * pred_boxes[i][4],
                  pred_boxes[i][2] - 0.5 * pred_boxes[i][5], 1]

        point8 = [pred_boxes[i][0] + 0.5 * pred_boxes[i][3], pred_boxes[i][1] - 0.5 * pred_boxes[i][4],
                  pred_boxes[i][2] - 0.5 * pred_boxes[i][5], 1]

        velo_box.append([point1, point2, point3, point4, point5, point6, point7, point8])

    velo_box = np.array(velo_box)

    for calib_line in kitti_calib_lines:
        if 'P2' in calib_line:
            P2 = calib_line.split(' ')[1:]
            P2 = np.array(P2, dtype='float').reshape(3, 4)
        elif 'R0_rect' in calib_line:
            R0_rect = np.zeros((4, 4))
            R0 = calib_line.split(' ')[1:]
            R0 = np.array(R0, dtype='float').reshape(3, 3)
            R0_rect[:3, :3] = R0
            R0_rect[-1, -1] = 1
        elif 'velo_to_cam' in calib_line:
            velo_to_cam = np.zeros((4, 4))
            velo2cam = calib_line.split(' ')[1:]
            velo2cam = np.array(velo2cam, dtype='float').reshape(3, 4)
            velo_to_cam[:3, :] = velo2cam
            velo_to_cam[-1, -1] = 1

    tran_mat = P2.dot(R0_rect).dot(velo_to_cam)  # 3x4

    velo_box = velo_box.reshape(-1, 4).T
    img_box = np.dot(tran_mat, velo_box).T
    img_box = img_box.reshape(-1, 8, 3)

    # img_box[:, :, 2] = np.abs(img_box[:, :, 2])

    img_box[:, :, 0] = img_box[:, :, 0] / (img_box[:, :, 2])
    img_box[:, :, 1] = img_box[:, :, 1] / (img_box[:, :, 2])


    return img_box, pred_dicts[:, 1]


def box2_calculator(pred_dicts):
    pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
    # pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
    # pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
    """
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        Args:
            boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    """

    velo_box = []

    for i in range(len(pred_boxes)):
        point1 = [pred_boxes[i][0] + 0.5 * pred_boxes[i][3], pred_boxes[i][1] + 0.5 * pred_boxes[i][4],
                  pred_boxes[i][2] + 0.5 * pred_boxes[i][5], 1]

        point2 = [pred_boxes[i][0] - 0.5 * pred_boxes[i][3], pred_boxes[i][1] + 0.5 * pred_boxes[i][4],
                  pred_boxes[i][2] + 0.5 * pred_boxes[i][5], 1]

        point3 = [pred_boxes[i][0] - 0.5 * pred_boxes[i][3], pred_boxes[i][1] - 0.5 * pred_boxes[i][4],
                  pred_boxes[i][2] + 0.5 * pred_boxes[i][5], 1]

        point4 = [pred_boxes[i][0] + 0.5 * pred_boxes[i][3], pred_boxes[i][1] - 0.5 * pred_boxes[i][4],
                  pred_boxes[i][2] + 0.5 * pred_boxes[i][5], 1]

        point5 = [pred_boxes[i][0] + 0.5 * pred_boxes[i][3], pred_boxes[i][1] + 0.5 * pred_boxes[i][4],
                  pred_boxes[i][2] - 0.5 * pred_boxes[i][5], 1]

        point6 = [pred_boxes[i][0] - 0.5 * pred_boxes[i][3], pred_boxes[i][1] + 0.5 * pred_boxes[i][4],
                  pred_boxes[i][2] - 0.5 * pred_boxes[i][5], 1]

        point7 = [pred_boxes[i][0] - 0.5 * pred_boxes[i][3], pred_boxes[i][1] - 0.5 * pred_boxes[i][4],
                  pred_boxes[i][2] - 0.5 * pred_boxes[i][5], 1]

        point8 = [pred_boxes[i][0] + 0.5 * pred_boxes[i][3], pred_boxes[i][1] - 0.5 * pred_boxes[i][4],
                  pred_boxes[i][2] - 0.5 * pred_boxes[i][5], 1]

        velo_box.append([point1, point2, point3, point4, point5, point6, point7, point8])

    velo_box = np.array(velo_box)

    for calib_line in kitti_calib_lines:
        if 'P2' in calib_line:
            P2 = calib_line.split(' ')[1:]
            P2 = np.array(P2, dtype='float').reshape(3, 4)
        elif 'R0_rect' in calib_line:
            R0_rect = np.zeros((4, 4))
            R0 = calib_line.split(' ')[1:]
            R0 = np.array(R0, dtype='float').reshape(3, 3)
            R0_rect[:3, :3] = R0
            R0_rect[-1, -1] = 1
        elif 'velo_to_cam' in calib_line:
            velo_to_cam = np.zeros((4, 4))
            velo2cam = calib_line.split(' ')[1:]
            velo2cam = np.array(velo2cam, dtype='float').reshape(3, 4)
            velo_to_cam[:3, :] = velo2cam
            velo_to_cam[-1, -1] = 1

    tran_mat = P2.dot(R0_rect).dot(velo_to_cam)  # 3x4

    velo_box = velo_box.reshape(-1, 4).T
    img_box = np.dot(tran_mat, velo_box).T
    img_box = img_box.reshape(-1, 8, 3)

    img_box[:, :, 0] = img_box[:, :, 0] / img_box[:, :, 2]
    img_box[:, :, 1] = img_box[:, :, 1] / img_box[:, :, 2]

    img_box = img_box[:, :, :2]  # （n,8,2）

    x1y1 = np.min(img_box, axis=1)
    x2y2 = np.max(img_box, axis=1)
    result = np.hstack((x1y1, x2y2))  # （n,4）
    return result
