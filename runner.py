import argparse
import logging
import torch
import os
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import NeRF, SDFNetwork, SingleVarianceNetwork, RenderingNetwork
from models.render import NeuSRender
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tqdm

class Runner:
    def __init__(self, conf_path, case,is_continue, mode='train'):
        # 运行模式，默认训练
        self.mode = mode
        # 指定运行设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cup")
        # 配置文件的路径
        self.conf_path = conf_path
        # 读取配置文件的内容
        f = open(self.conf_path)
        conf_text = f.read()
        # CASE_NAME在配置文件的作用可以理解为占位的字符串,因此这里用case中的内容进行了替换
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()
        # 将配置内容的格式变换成树形结构的形式
        self.conf = ConfigFactory.parse_string(conf_text)
        # 训练(指定)结果的存放位置
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)

        # 初始化一个data数据类
        self.dataset = Dataset(self.conf['dataset'])
        # -----训练参数设置-----
        # 开始训练的迭代epoch序号
        self.iter_step = 0
        # 结束训练的迭代epoch序号
        self.end_iter = self.conf.get_int('train.end_iter')
        # 训练过程中保存模型权重的周期
        self.save_freq = self.conf.get_int('train.save_freq')
        # 训练过程中打印必要信息的周期(loss和学习率)
        self.report_freq = self.conf.get_int('train.report_freq')
        # 训练过程中合成一个rgb视角图的周期
        self.val_freq = self.conf.get_int('train.val_freq')
        # 训练过程中生成一个ply模型的周期
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        # 训练过程中的batchsize(rays的个数)
        self.batch_size = self.conf.get_int('train.batch_size')
        # 理解成图片下采样的倍数
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        # 学习率
        self.learning_rate = self.conf.get_float('train.learning_rate')
        # 控制学习率变化的参数
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        # 是否使用白色背景
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        # 预热启动区间
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        # 退火区间
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        # -----训练参数设置-----

        # -----neus网络模型设置-----
        # 计算loss时,sdf的梯度loss占整个loss的权重
        self.igr_weight = self.conf.get_float('train.igr_weight')
        # 计算loss时,mask的loss占整个loss的权重
        self.mask_weight = self.conf.get_float('train.mask_weight')
        # 是否在已有的最新模型基础上进行下一步操作
        self.is_continue = is_continue
        self.model_list = []
        # 用于存放所有神经网络模型的参数
        params_to_train = []
        # nerf网络
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        # sdf网络
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        # 偏差网络
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        # 渲染网络
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)

        #TODO :打印+=的区别 即 list前后的type
        params_to_train.append(list(self.nerf_outside.parameters()))
        params_to_train.append(list(self.sdf_network.parameters()))
        params_to_train.append(list(self.deviation_network.parameters()))
        params_to_train.append(list(self.color_network.parameters()))

        # 设置优化器
        self.optimizer = torch.optim.Adam(params_to_train,lr=self.learning_rate)

        # 初始化neus神经网络
        self.renderer = NeuSRender(self.nerf_outside,
                                   self.sdf_network,
                                   self.deviation_network,
                                   self.color_network,
                                   **self.conf['model.neus_renderer'])
        # 加载断点权重
        latest_model_name = None

        # 选择最新模型
        if is_continue:
            # 加载目录下的所有文件
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir,'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == "pth" and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
        # 对权重文件进行排序
        model_list.sort()
        latest_model_name = model_list[-1]

        # 若存在权重，则加载
        if latest_model_name is not None:
            logging.info("Find checkpoint {}".format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # 训练模式下，复制重要py文件到exp中
        if self.mode[:5] == "train":
            self.file_backup()
    def load_checkpoint(self,ckeckpoint_name):
        checkpoint =torch.load(os.path.join(self.base_exp_dir,'checkpoints',ckeckpoint_name))
        # 加载各网络模块
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])

        # 加载优化器
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # 加载开始迭代序号
        self.iter_step = checkpoint['iter_step']
        logging.info("load checkpoint end")

    def file_backup(self):
        # 指定拷贝目标文件所在目录
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir,'recording'),exist_ok=True)

        # 拷贝特定文件
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir,"recording",dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyright(os.path.join(dir_name,f_name),os.path.join(cur_dir,f_name))
        copyright(self.conf,os.path.join(self.base_exp_dir,'recoding','config.conf'))




        print("backup")

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        #  sort images randomly
        image_perm = self.get_image_perm()
        for iter_i in tqdm(range(res_step)):
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)
            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3:6], data[:, 6:9], data[:, 9:10]
            near, far = self.dataset.near_far_from_sphere(rays_o,rays_d)
    def update_learning_rate(self):
        # warm start
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step/self.warm_up_end

        # stand strat
        else:
            alpha = self.learning_rate_alpha
            # training progress
            progress = (self.iter_step - self.warm_up_end)/(self.end_iter - self.warm_up_end)
            # learning_factor  range = (1-alpha-1)
            learning_factor = np.cos((np.pi * progress)+1.0) * 0.5 * (1 - alpha) + alpha
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def get_image_perm(self):
        # generate n_images index
        return torch.randperm(self.dataset.n_images)


if __name__ == '__main__':
    print("start main ")

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--conf',type=str,default='./conf/base.conf')
    # parser.add_argument('--mode',type=str,default='train')  # render/train/fine
    # parser.add_argument('--mcube_threshold',type=float,default=0.0)  # threshold value
    # parser.add_argument('--is_continue', type=bool, default=False,action="store_true")
    # parser.add_argument('--gpu', type=int, default=0)
    # parser.add_argument('--case', type=str, default="")  # dataset class filesname
    # args = parser.parse_args()





# t = Runner('conf/womask.conf','aaa')
object_bbox_min = np.array([[-1.01, -1.01, -1.01, 1.0],
                           [-2, -1.3, -1.01, 1.0],
                           [-21, -31, -1.01, 11],
                           [-1.5, -14, -1.04, 1.0]])
# print(object_bbox_min[:,])
lines = []
# lines = [[x[0], x[1], x[2], x[3]] for x in object_bbox_min]
# print(torch.randperm(10))
# a = object_bbox_min[None, :2, :1]
lines.append(lambda x: x)
print(lines)