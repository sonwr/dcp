import os
import gc
import argparse
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import open3d as o3d
from torch.optim.lr_scheduler import MultiStepLR
from data import ModelNet40
from model_w import DCP
from util import transform_point_cloud, npmat2euler
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm


def visualize_pointcloud(pointcloud1, pointcloud2):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pointcloud1)
    pcd1.paint_uniform_color([1, 0, 0])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pointcloud2)
    pcd2.paint_uniform_color([0, 0, 1])
    
    o3d.visualization.draw_geometries([pcd1, pcd2])


def test_one_index(args, net, test_loader, index):
    net.eval()
    
    src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba = test_loader.dataset[index]

    src = torch.tensor(src).unsqueeze(0).cuda()
    target = torch.tensor(target).unsqueeze(0).cuda()

    # cpu -> cuda
    rotation_ab = torch.tensor(rotation_ab).unsqueeze(0).cuda()
    rotation_ba = torch.tensor(rotation_ba).unsqueeze(0).cuda()
    translation_ab = torch.tensor(translation_ab).unsqueeze(0).cuda()
    translation_ba = torch.tensor(translation_ba).unsqueeze(0).cuda()

    # net
    rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src, target)

    # cuda -> cpu
    rotation_ab = rotation_ab.detach().cpu().numpy()
    rotation_ba = rotation_ba.detach().cpu().numpy()
    translation_ab = translation_ab.detach().cpu().numpy()
    translation_ba = translation_ba.detach().cpu().numpy()

    rotation_ab_pred = rotation_ab_pred.detach().cpu().numpy()
    translation_ab_pred = translation_ab_pred.detach().cpu().numpy()
    rotation_ba_pred = rotation_ba_pred.detach().cpu().numpy()
    translation_ba_pred = translation_ba_pred.detach().cpu().numpy()

    # euler_ba
    # Transform point clouds
    transformed_src = transform_point_cloud(src, torch.tensor(rotation_ab_pred).cuda(), torch.tensor(translation_ab_pred).cuda())
    transformed_src_np = transformed_src.detach().cpu().numpy().squeeze().transpose(1, 0)

    target_np = target.detach().cpu().numpy().squeeze().transpose(1, 0)

    # Visualize transformed_src with target
    visualize_pointcloud(transformed_src_np, target_np)  
    


def main():
    class Args:
        exp_name = 'dcp_v1'
        model = 'dcp'
        emb_nn = 'pointnet'
        pointer = 'identity'
        head = 'svd'
        emb_dims = 512
        n_blocks = 1
        n_heads = 4
        ff_dims = 1024
        dropout = 0.0
        batch_size = 32
        test_batch_size = 10
        epochs = 5
        use_sgd = False
        lr = 0.001
        momentum = 0.9
        no_cuda = False
        seed = 1234
        cycle = False
        gaussian_noise = False
        unseen = False
        num_points = 1024
        dataset = 'modelnet40'
        factor = 4
        model_path = ''
        index_to_test = 1  # New argument for specifying the index to test

    args = Args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'modelnet40':
        train_loader = DataLoader(
            ModelNet40(num_points=args.num_points, partition='train', gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            ModelNet40(num_points=args.num_points, partition='test', gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        raise Exception("not implemented")

    if args.model == 'dcp':
        net = DCP(args).cuda()
        
        if args.model_path == '':
            model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
        else:
            model_path = args.model_path
            print(model_path)
        if not os.path.exists(model_path):
            print("can't find pretrained model")
            return
        net.load_state_dict(torch.load(model_path), strict=False)
        
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        raise Exception('Not implemented')

    test_one_index(args, net, test_loader, args.index_to_test)

    #textio.cprint('==TESTING ONE INDEX==')
    #textio.cprint('Rotation AB: {}'.format(rotations_ab))
    #textio.cprint('Translation AB: {}'.format(translations_ab))
    #textio.cprint('Predicted Rotation AB: {}'.format(rotations_ab_pred))
    #textio.cprint('Predicted Translation AB: {}'.format(translations_ab_pred))
    #textio.cprint('Rotation BA: {}'.format(rotations_ba))
    #textio.cprint('Translation BA: {}'.format(translations_ba))
    #textio.cprint('Predicted Rotation BA: {}'.format(rotations_ba_pred))
    #textio.cprint('Predicted Translation BA: {}'.format(translations_ba_pred))

    print('FINISH')

if __name__ == '__main__':
    main()
