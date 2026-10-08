from data.nyu_dataloader import *
from data.middlebury_dataloader import *
from models.model import *
from utils import *
import argparse
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default= 16, help='scale factor')
    parser.add_argument("--num_feats", type=int, default = 32, help='channel number of the middle hidden layer')
    parser.add_argument("--root_dir", type=str, default='Lu', help="root dir of testdataset")
    parser.add_argument("--model_dir", type=str, default='best_model_x16.pth', help="root dir of model")
    parser.add_argument("--result_dir", type=str, default='result', help="root dir of result")

    opt = parser.parse_args()

    net = Mynet(num_feats=opt.num_feats, scale=opt.scale, kernel_size=3)
    net.load_state_dict(torch.load(opt.model_dir, map_location='cuda:0'))
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    net.to(device)

    data_transform = transforms.Compose([transforms.ToTensor()])

    dataset_name = opt.root_dir.split('/')[-1]
    if dataset_name == 'nyu_data':
        dataset = NYU_v2_dataset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform, train=False)
        test_minmax = np.load('%s/test_minmax.npy'% opt.root_dir)
        rmse = np.zeros(449)
    elif dataset_name == 'Middlebury':
        dataset = Middlebury_dataset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform)
        rmse = np.zeros(30)
    elif dataset_name == 'Lu':
        dataset = Middlebury_dataset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform)
        rmse = np.zeros(6)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    data_num = len(dataloader)


    with torch.no_grad():
        net.eval()
        if dataset_name == 'nyu_data':
            for idx, data in enumerate(dataloader):
                guidance, lr, gt = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(device)
                out = net((guidance, lr))
                minmax = test_minmax[:, idx]
                minmax = torch.from_numpy(minmax).cuda()
                rmse[idx] = calc_rmse(gt[0, 0], out[0, 0], minmax)

                path_output = '{}/output'.format(opt.result_dir)
                os.makedirs(path_output, exist_ok=True)


                path_save_pred = '{}/{:010d}.png'.format(path_output, idx)
                pred_metric = out[0, 0] * (minmax[0] - minmax[1]) + minmax[1]
                pred_mm = pred_metric * 1000.0
                pred_mm_np = pred_mm.cpu().detach().numpy()
                pred_save = pred_mm_np.astype(np.uint16)
                Image.fromarray(pred_save).save(path_save_pred)
                path_save_vis = '{}/{:010d}_vis.png'.format(path_output, idx)
                pred_np = out[0, 0].cpu().detach().numpy()
                gt_np = gt[0, 0].cpu().detach().numpy()
                v_min = min(pred_np.min(), gt_np.min())
                v_max = max(pred_np.max(), gt_np.max())
                pred_u8 = ((pred_np - v_min) / (v_max - v_min + 1e-8) * 255.0).astype(np.uint8)
                gt_u8 = ((gt_np - v_min) / (v_max - v_min + 1e-8) * 255.0).astype(np.uint8)
                pred_color = cv2.applyColorMap(pred_u8, cv2.COLORMAP_JET)
                gt_color = cv2.applyColorMap(gt_u8, cv2.COLORMAP_JET)
                combined = np.hstack([gt_color])
                cv2.imwrite(path_save_vis, combined)

                print(f"NYU Image: {idx} | RMSE: {rmse[idx]}")
            print("Mean RMSE:", rmse.mean())

        elif (dataset_name == 'Middlebury') or (dataset_name == 'Lu'):
            for idx, data in enumerate(dataloader):
                guidance, lr, gt = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(device)
                out = net((guidance, lr))
                rmse[idx] = midd_calc_rmse(gt[0, 0], out[0, 0])

                path_output = '{}/output'.format(opt.result_dir)
                os.makedirs(path_output, exist_ok=True)

                path_save_pred = '{}/{:010d}.png'.format(path_output, idx)
                pred_np = out[0, 0].cpu().detach().numpy()


                pred_gray = (pred_np * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(pred_gray).save(path_save_pred)


                path_save_vis = '{}/{:010d}_vis.png'.format(path_output, idx)


                gt_np = gt[0, 0].cpu().detach().numpy()


                pred_u8 = (pred_np * 255.0).clip(0, 255).astype(np.uint8)
                gt_u8 = (gt_np * 255.0).clip(0, 255).astype(np.uint8)


                pred_color = cv2.applyColorMap(pred_u8, cv2.COLORMAP_JET)
                gt_color = cv2.applyColorMap(gt_u8, cv2.COLORMAP_JET)


                combined = np.hstack([pred_color])


                cv2.imwrite(path_save_vis, combined)

                print(f"Midd Image: {idx} | RMSE: {rmse[idx]}")
            print("Mean RMSE:", rmse.mean())
