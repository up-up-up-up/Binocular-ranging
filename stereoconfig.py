import numpy as np

# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = np.array([   [479.6018,   -0.0769,  652.6060],
                                            [       0,  478.0229,  352.3870],
                                            [       0,         0,         1]
                                        ])
        # 右相机内参
        self.cam_matrix_right = np.array([  [489.9354,   0.2789,  641.6219],
                                            [       0,  487.9356,  354.5612],
                                            [       0,         0,         1]
                                        ])

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[-0.0791, 0.0309, -0.0009, -0.0004, -0.0091]])
        self.distortion_r = np.array([[-0.1153, 0.1021, -0.0011,  -0.0005,  -0.0459]])

        # 旋转矩阵
        self.R = np.array([ [1.0000,  0.0005, -0.0184],
                            [-0.0005,  1.0000, 0.0001],
                            [ 0.0184,  -0.0001,  1.0000]
                            ])
        # 平移矩阵
        self.T = np.array([[121.4655], [0.2118], [0.5950]])
        # 焦距
        self.focal_length = 749.402 # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]
        # 基线距离
        self.baseline = 121.4655 # 单位：mm， 为平移向量的第一个参数（取绝对值）