import os.path

import numpy as np
import torch
import numpy as np
import os
import glob
import cv2 as cv


def load_K_Rt_from_P(filename,P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
            lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()
    # resolve P to intri K and Extri Rt
    out = cv.decomposeProjectionMatrix(P)
    K = out[0]  # intri
    R = out[1]  # Extri rotate
    t = out[2]  # Extir translation
    K = K/K[2, 2]
    # intrinsics (4x4)
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K
    # Extrinsics (4X4)
    pose = np.eye(4,dtype=np.float32)
    # T
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3]/t[3])[:, 0]
    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        # 配置文件
        self.conf = conf
        # 数据存放路径
        self.data_dir = conf.get_string('data_dir')
        # 相机投影矩阵
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        # camera params whether include camera_outside_sphere
        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        # whether include scale_mat_scale
        self.scale_mat_scale = conf.get_float('scale_mat_scale',default=1.1)

        # load camera sphere
        camera_dict = np.load(os.path.join(self.data_dir,self.render_cameras_name))
        self.camera_dict = camera_dict

        # load images
        self.images_list = sorted(glob(os.path.join(self.data_dir,'image/*.png')))
        self.n_images = len(self.images_list)
        # normalization
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_list])/256.0
        # mask images
        self.masks_list = sorted(glob(os.path.join(self.data_dir,'mask/*.png')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_list])/256.0
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).to(self.device)
        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).to(self.device)
        # image size
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        # pixels
        self.image_pixels = self.H * self.W

        # images coordinate to world coordinate 4X4
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        # normalization renderer scence in the unit ball
        self.scale_mats_np = []
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        # images correspondence intrinsics
        self.intrinsics_all = []
        # images correspondence extrinsics/pose
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np,self.world_mats_np):
            # scale mat sphere
            P = world_mat @ scale_mat
            P = P[:3, :4]  # delete last layer [0,0,0,1]
            # get Extrinsices and Intrinsics in camera sphere
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        self.pose_all = torch.stack(self.pose_all).to(self.device)
        # focal
        self.focal = self.intrinsics_all[0][0, 0]

        # coordinate mapping
        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
        object_scale_mat = np.load(os.path.join(self.data_dir,self.object_cameras_name))['scale_mat_0']

        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        genrate random rays at world space from one camera(image)
        :param img_idx:
        :param batch_size:
        :return:
        """
        # get batch_size pixels randomly
        pixels_x = torch.randint(low=0,high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0,high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_x,pixels_y)]
        mask = self.masks[img_idx][(pixels_x,pixels_y)]

        # camera coordinate ray [batch_size,3]
        p = torch.stack([pixels_x,pixels_y,torch.ones_like(pixels_x)],dim=-1).float()
        p = torch.matmul(self.intrisics_all_inv[img_idx,None,:3,:3],p[:,:,None]).squeeze()
        # normalization [batch_size,3]
        rays_v = p / torch.linalg.norm(p,ord=2,dim=-1,keepdim=True)

        # world coordinate ray direction= camera X R of Extri
        rays_v = torch.matmul(self.pose_all[img_idx,None,:3,:3],rays_v[:,:,None]).squeeze()
        # world coordinate ray origin  = t of Extri
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)

        return torch.cat([rays_o.to(self.device), rays_v.to(self.device),color,mask[:,:,1]],dim=-1).cuda()

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)

        b = 2.0 * torch.sum(rays_o*rays_d,dim=-1,keepdim=True)

        mid = 0.5 * (-b) / a

        near = mid - 1.0
        far = mid + 1.0
        return near, far











