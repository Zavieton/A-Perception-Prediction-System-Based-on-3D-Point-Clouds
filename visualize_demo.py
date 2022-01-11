import mayavi.mlab as mlab
from visual_utils import visualize_utils as V
from pathlib import Path
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu, load_data_to_cpu
from pcdet.utils import common_utils
from box_calculator import box_calculator, point_pcl_to_image
import os
from my_visualize import draw_projected_box3d, draw_predect
import cv2
from pcdet.datasets import DatasetTemplate
import numpy as np
import glob
from my_ab3dmot import AB3DMOT_fun
from TrajNet import traject, traject2
from AB3DMOT_libs.model import AB3DMOT
import individual_TF

#定义参数，transformer
device=torch.device("cuda")
emb_size = 512
num_samples = 20
layers = 6
heads = 8
dropout = 0.1

# pointpillars类
class PointpillarsDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict



def predict(cfg_file, data_path_in, ckpt, ext, ID):

    data_path = os.path.join(data_path_in, 'velodyne', ID)  #读取点云文件
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger()
    demo_dataset = PointpillarsDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(data_path), ext=ext, logger=logger
    ) # 创建dataset

    print(f'Total number of samples: \t{len(demo_dataset)}')


    show_pointpilalrs = False # opencv显示
    ab3dmot = True # 轨迹跟踪
    mot_tracker = AB3DMOT() # 创建跟踪器

    # pointpillars model init
    pointpillars = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    pointpillars.load_params_from_file(filename=ckpt, logger=logger, to_cpu=True)
    pointpillars.cuda()
    pointpillars.eval()

    # transformer model init
    transformer_models = individual_TF.IndividualTF(2, 3, 3, N=layers,
                                       d_model=emb_size, d_ff=2048, h=heads, dropout=dropout,
                                       mean=[0, 0], std=[0, 0]).to(device)

    transformer_models.load_state_dict(torch.load(f'/home/zavieton/3D/pedestrian_predict/Trajectory-Transformer/models/Individual/inte_xz/00185.pth'))

    transformer_models.to(device)

    trajectorys = []

    label_occured = []

    # 主要循环，逐帧判断
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            print(f'Visualized  index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict) # 加载数据到gpu
            pred_dicts, _ = pointpillars.forward(data_dict) # pointpillars预测输出

            if ab3dmot: # AB3DMOT
                res = AB3DMOT_fun(pred_dicts, idx, mot_tracker)   # res: frame id x y z h w l
                res_to_trajnet = res[:, 0:5].reshape(-1, 5) # res取ab3dmot跟踪结果的前五位，即[frame id x y z]
                if idx == 0:
                    to_transformer = res_to_trajnet # 第一帧，创建一个array， 便于后续append
                else:
                    to_transformer = np.concatenate((to_transformer, res_to_trajnet), axis=0) # axis=0  array向下append
                    # if idx % 8 == 0:
                    if idx >= 48 and idx % 1 == 0: # 当frame 大于 obs 时开始预测
                        trajectorys, peds = traject2(to_transformer, transformer_models) # 输出为预测的‘preds’个轨迹点，和人的ID

            if idx==80:
                fig = V.draw_scenes(
                    points=data_dict['points'][:, 1:], ref_boxes=res[:, 2:9].reshape(-1, 7), ref_id=res[:, 1]
                )

                # print(trajectorys)
                # trajectorys = -trajectorys
                for b in trajectorys:
                    b = b.reshape(-1, 3)
                    mlab.points3d(b[:, 0], b[:, 2], b[:, 1], line_width=0.25, figure=fig, color=(1, 0, 0), scale_factor=0.15)

                mlab.show()
            '''
            if idx >= 20:
                traj_furture = res_to_trajnet[:, 2:]
                print(traj_furture)
                for b in traj_furture:
                    b = b.reshape(-1, 3)
                    mlab.points3d(b[:, 0], b[:, 1], b[:, 2], line_width=0.25, figure=fig, color=(0, 1, 0), scale_factor=0.15)


            if idx == 20+32:
                mlab.show()'''

def predict_feedback(cfg_file, data_path_in, ckpt, ext, ID):

    data_path = os.path.join(data_path_in, 'velodyne', ID)  #读取点云文件
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger()
    demo_dataset = PointpillarsDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(data_path), ext=ext, logger=logger
    ) # 创建dataset

    print(f'Total number of samples: \t{len(demo_dataset)}')


    show_pointpilalrs = False # opencv显示
    ab3dmot = True # 轨迹跟踪
    mot_tracker = AB3DMOT() # 创建跟踪器

    # pointpillars model init
    pointpillars = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    pointpillars.load_params_from_file(filename=ckpt, logger=logger, to_cpu=True)
    pointpillars.cuda()
    pointpillars.eval()

    # transformer model init
    transformer_models = individual_TF.IndividualTF(2, 3, 3, N=layers,
                                       d_model=emb_size, d_ff=2048, h=heads, dropout=dropout,
                                       mean=[0, 0], std=[0, 0]).to(device)

    transformer_models.load_state_dict(torch.load(f'/home/zavieton/3D/pedestrian_predict/Trajectory-Transformer/models/Individual/inte_xz/00185.pth'))

    transformer_models.to(device)

    trajectorys = []
    label_disappear_recently = []
    position_disappear_recently = []
    front_label = []
    front_position = []

    # 主要循环，逐帧判断
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            print(f'Visualized  index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict) # 加载数据到gpu
            pred_dicts, _ = pointpillars.forward(data_dict) # pointpillars预测输出

            if ab3dmot: # AB3DMOT
                res = AB3DMOT_fun(pred_dicts, idx, mot_tracker)   # res: frame id x y z h w l
                res_to_trajnet = res[:, 0:5].reshape(-1, 5) # res取ab3dmot跟踪结果的前五位，即[frame id x y z]
                if idx == 0:
                    to_transformer = res_to_trajnet # 第一帧，创建一个array， 便于后续append
                else:
                    to_transformer = np.concatenate((to_transformer, res_to_trajnet), axis=0) # axis=0  array向下append
                    # if idx % 8 == 0:
                    if idx >= 30 : # 当frame 大于 obs 时开始预测
                        trajectorys, peds = traject2(to_transformer, transformer_models) # 输出为预测的‘preds’个轨迹点，和人的ID


                # feedback
                label = res_to_trajnet[:, 1]
                current_add = []
                current_disappear = []
                # print(res_to_trajnet)
                if front_label == []:
                    pass
                else:
                    for i in range(len(label)):
                        if label[i] not in front_label:
                            current_add.append(label[i])

                    for i in range(len(front_label)):
                        if front_label[i] not in label:
                            current_disappear.append(front_label[i])

                if current_disappear != []:
                    label_disappear_recently.append([current_disappear]) #
                    for each in current_disappear:
                        position_disappear_recently.append(front_position[front_position[:, 0] == each])


                if len(label_disappear_recently) > 30:
                    del label_disappear_recently[0]
                    del position_disappear_recently[0]

                # print(position_disappear_recently)

                if current_add != []:
                    for each in current_add:
                        pos = res_to_trajnet[res_to_trajnet[:, 1] == each]
                        position = pos[:, 2:][0]

                        for j in position_disappear_recently:
                            # print((position[0] - j[0][1])**2 + (position[1] - j[0][2])**2 + (position[2] - j[0][3])**2)

                            if ((position[0] - j[0][1])**2 + (position[1] - j[0][2])**2 + (position[2] - j[0][3])**2) < 10:
                                # print(j[0][0])
                                # print(res_to_trajnet)

                                for k in res_to_trajnet:
                                    if k[1] == each:
                                        k[1] = j[0][0]
                                        break
                                # res_to_trajnet[res_to_trajnet[:, 1] == each][0][1] = j[0][0]
                                for k in res:
                                    if k[1] == each:
                                        k[1] = j[0][0]
                                    break
                                break


                front_label = label
                front_position = res_to_trajnet[:, 1:]
                # end feedback


            if  idx == 80:
                fig = V.draw_scenes(
                    points=data_dict['points'][:, 1:], ref_boxes=res[:, 2:9].reshape(-1, 7), ref_id=res[:, 1]
                )

                # print(trajectorys)
                # trajectorys = -trajectorys
                for b in trajectorys:
                    b = b.reshape(-1, 3)
                    mlab.points3d(b[:, 0], b[:, 2], b[:, 1], line_width=0.25, figure=fig, color=(1, 0, 0), scale_factor=0.15)

                mlab.show()
            '''
            if idx >= 20:
                traj_furture = res_to_trajnet[:, 2:]
                print(traj_furture)
                for b in traj_furture:
                    b = b.reshape(-1, 3)
                    mlab.points3d(b[:, 0], b[:, 1], b[:, 2], line_width=0.25, figure=fig, color=(0, 1, 0), scale_factor=0.15)


            if idx == 20+32:
                mlab.show()'''

def main():

    cfg_file = 'cfgs/kitti_models/pointpillar.yaml'
    # cfg_file = 'cfgs/kitti_models/pointrcnn.yaml'

    data_path = '/home/zavieton/3D/pedestrian_predict/AB3DMOT/data/KITTI/resources/testing/'
    idx = '0023'

    ckpt = '/home/zavieton/3D/pedestrian_predict/OpenPCDet/pretrain/pointpillar_7728.pth'
    # ckpt = '/home/zavieton/3D/pedestrian_predict/OpenPCDet/pretrain/pointrcnn_7870.pth'

    ext = '.bin'
    predict(cfg_file, data_path, ckpt, ext, idx)  # 主要函数， pointpillars+Ab3dmot+transformer


if __name__ == '__main__':
    main()
