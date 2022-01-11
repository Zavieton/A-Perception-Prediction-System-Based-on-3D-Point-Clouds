import predict
import predict_feedback


def main():
    cfg_file = 'cfgs/kitti_models/pointpillar.yaml'
    # cfg_file = 'cfgs/kitti_models/pointrcnn.yaml'

    data_path = '/home/zavieton/3D/pedestrian_predict/AB3DMOT/data/KITTI/resources/testing/'
    idx = '0023'

    ckpt = 'pretrain_models/pointpillar_7728.pth'
    # ckpt = 'pretrain_models/pointrcnn_7870.pth'
    ext = '.bin'
    predict.predict(cfg_file, data_path, ckpt, ext, idx)  # 主要函数， pointpillars+Ab3dmot+transformer



if __name__ == "__main__":
    main()
