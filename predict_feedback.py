from pathlib import Path
# import mayavi.mlab as mlab
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
import time


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

# 主要函数
def predict(cfg_file, data_path_in, ckpt, ext, ID):
    data_path = os.path.join(data_path_in, 'velodyne', ID)  #读取点云文件
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger()
    demo_dataset = PointpillarsDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(data_path), ext=ext, logger=logger
    ) # 创建dataset

    print(f'Total number of samples: \t{len(demo_dataset)}')


    show_pointpilalrs = True # opencv显示
    ab3dmot = True # 轨迹跟踪
    mot_tracker = AB3DMOT() # 创建跟踪器
    if show_pointpilalrs:
        cv2.namedWindow('RESULT') # 创建窗口

    # pointpillars model init
    pointpillars = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    pointpillars.load_params_from_file(filename=ckpt, logger=logger, to_cpu=True)
    pointpillars.cuda()
    pointpillars.eval()

    # transformer model init
    transformer_models = individual_TF.IndividualTF(2, 3, 3, N=layers,
                                       d_model=emb_size, d_ff=2048, h=heads, dropout=dropout,
                                       mean=[0, 0], std=[0, 0]).to(device)

    transformer_models.load_state_dict(torch.load(f'pretrain_models/00235.pth'))

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
            start = time.time()
            pred_dicts, _ = pointpillars.forward(data_dict) # pointpillars预测输出
            end = time.time()
            # print("pointpillars using time:",end-start)

            if ab3dmot: # AB3DMOT
                start = time.time()
                res = AB3DMOT_fun(pred_dicts, idx, mot_tracker)   # res: frame id x y z h w l
                end = time.time()
                # print("ab3dmot:", end-start)
                # res_to_trajnet = np.concatenate((res[:, 0:3].reshape(-1, 3), res[:, 4].reshape(-1, 1)), axis=1)
                res_to_trajnet = res[:, 0:5].reshape(-1, 5) # res取ab3dmot跟踪结果的前五位，即[frame id x y z]
                res_to_trajnet[:, 2:] = -res_to_trajnet[:, 2:] # 加负号是因为pointpillars检测结果和训练集对应符号相反，可以不进行此操作应该


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




                if idx == 0:
                    to_transformer = res_to_trajnet # 第一帧，创建一个array， 便于后续append
                else:
                    to_transformer = np.concatenate((to_transformer, res_to_trajnet), axis=0) #axis=0  array向下append
                    # if idx % 8 == 0:
                    if idx >= 40 and idx % 1 == 0: # 当frame 大于 obs 时开始预测
                        start = time.time()
                        trajectorys, peds = traject2(to_transformer, transformer_models) #输出为预测的‘preds’个轨迹点，和人的ID
                        end = time.time()
                        # print("transfomer:",end-start)

                        # print(trajectorys)
                        # print(trajectorys, peds)
                        # print("res", res[:, 2:5])
                        # print("trajectorys", trajectorys)
                        # print(trajectorys)

            if show_pointpilalrs: # opencv显示结果
                img = cv2.imread(os.path.join(data_path_in, 'image_02', ID, str(idx).zfill(6)+'.png')) #从文件夹读取图片
                dim3box, labels = box_calculator(res) # box_calculator 求3D框和标签

                if trajectorys != []: 
                    points = point_pcl_to_image(trajectorys) # pcl坐标转化为像素坐标
                    img = draw_predect(img, points)

                for i in range(dim3box.shape[0]):
                    img = draw_projected_box3d(img, dim3box[i], labels[i])

                cv2.imshow("RESULT", img)
                cv2.waitKey(1)



