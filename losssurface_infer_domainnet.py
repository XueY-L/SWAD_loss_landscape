"""Make Loss surface plane ((w1, w2, w3)-plane) and infer the grids.
https://github.com/timgaripov/dnn-mode-connectivity/blob/master/plane.py

CUDA_VISIBLE_DEVICES=1 python3 losssurface_infer_domainnet.py --src1 painting --src2 sketch --src3 real --tar clipart
"""
import argparse
import copy
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels
import numpy as np
from tqdm.auto import tqdm
from munch import Munch

from domainnet import DomainNetLoader
from lerp import lerp_multi
import utils

from logger import Logger

logger = Logger.get()


def params_to_vector(parameters):
    return torch.cat(list(map(lambda x: x.detach().flatten(), parameters)))


def get_xy(point, origin, vector_x, vector_y):
    return torch.as_tensor(
        [
            torch.dot(point - origin, vector_x),
            torch.dot(point - origin, vector_y)
        ]
    )


def get_basis(w1, w2, w3):
    """https://github.com/timgaripov/dnn-mode-connectivity/blob/master/plane.py#L105

    Args:
        w1, w2, w3: 1-dim torch tensor (vector)
    """
    u = w2 - w1
    du = u.norm()  # 求2范数
    u /= du

    v = w3 - w1
    print(u.dot(v))
    v -= u.dot(v) * u  # dot矩阵乘法，*按位乘
    dv = v.norm()
    v /= dv

    return u, v, du, dv


def copy_flat_params_(flat_params, model):
    offset = 0
    for p in model.parameters():
        size = p.numel()
        ip = flat_params[offset:offset+size].view(p.shape)
        with torch.no_grad():
            p.copy_(ip)
        offset += size


def infer_grid(w1, w2, w3, base_model, train_loader, test_loader, G, margin=0.2, update_bn=False):
    """Make a grid by (w1, w2, w3)-plane and infer for each grid point.
    https://github.com/timgaripov/dnn-mode-connectivity/blob/master/plane.py

    Args:
        w1, w2, w3: 1-dim torch tensor (vector)
        base_model: model for architecture.
        test_loader: dataloader for test env.
        G: n_grid_points (per axis); total points = G * G.
        margin
    """
    u, v, du, dv = get_basis(w1, w2, w3)
    print(w1)
    print(w2)
    print(w3)
    print(u, du)

    alphas = np.linspace(0. - margin, 1. + margin, G)
    betas = np.linspace(0. - margin, 1. + margin, G)

    results = []

    for i, w in enumerate([w1, w2, w3]):
        c = get_xy(w, w1, u, v)

        results.append({
            "ij": f"w{i+1}",
            "grid_xy": c
        })

    tk = tqdm(total=G*G)
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            tk.set_description(f"i={i+1}/{G}, j={j+1}/{G}")
            interpolated = w1 + alpha * du * u + beta * dv * v
            copy_flat_params_(interpolated, base_model)
            
            # update_bn
            if update_bn:
                utils.update_bn(train_loader, base_model)

            # inference loss & accuracy
            base_model.eval()
            l, correct = 0, 0
            with torch.no_grad():
                for batch_idx, (data, label) in enumerate(test_loader):
                    data, label = data.cuda(), label.cuda()
                    rst = base_model(data)

                    l += F.cross_entropy(rst, label).item() * label.size(0)
                    _, predicted = torch.max(rst.data, 1)
                    correct += predicted.eq(label.data).cpu().sum()
                    # print(rst, predicted, correct)
                    
            acc = correct / 21580  # 测试集样本数量
            loss = l / 21580       # 测试集样本数量
            print(acc, loss)

            #  c = get_xy(interpolated, w1, u, v)
            #  c == [alpha * dx, beta * dy] -> it has a little residual < 0.01.

            results.append({
                "ij": [i, j],
                "grid_xy": torch.as_tensor([alpha * du, beta * dv]),
                "error": 1. - acc,
                "loss": loss
            })
            tk.update(1)

    return results


def run(args, G=None, margin=0.6):
    """
    Args:
        G: # of ticks for each axis
        margin: horizontal & vertical margin
        mode: 'test' (test-in only) / 'train' (train in/out) / 'all'
        first, middle, last: checkpoint index
    """
    if G is None:
        G = int((1.0 + margin*2) * 15)
        logger.info(f"G = {G}")

    # load checkpoints
    logger.info("# Load checkpoints ...")
    model_path1 = f'/home/xingxuanzhang/yuanxue/loss-landscape/model/ckpt_{args.src1}__sgd_lr-s0.001_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth'
    model_path2 = f'/home/xingxuanzhang/yuanxue/loss-landscape/model/ckpt_{args.src2}__sgd_lr-s0.001_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth'
    model_path3 = f'/home/xingxuanzhang/yuanxue/loss-landscape/model/ckpt_{args.src3}__sgd_lr-s0.001_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth'

    param1 = torch.load(model_path1)['net']
    param2 = torch.load(model_path2)['net']
    try:
        param3 = torch.load(model_path3)['net']
    except:
        param_f = lerp_multi([param1, param2], [args.weight_src1, 1-args.weight_src1])

    net1 = tmodels.resnet50(num_classes=345)
    net2 = tmodels.resnet50(num_classes=345)
    net3 = tmodels.resnet50(num_classes=345)

    net1.load_state_dict(param1)
    net2.load_state_dict(param2)
    try:
        net3.load_state_dict(param3)
    except:
        net3.load_state_dict(param_f)

    w1 = params_to_vector(net1.parameters())
    w2 = params_to_vector(net2.parameters())
    w3 = params_to_vector(net3.parameters())

    base_model = copy.deepcopy(net1).cuda()

    # Build test dataloader
    trainloader, _, testloader = DomainNetLoader(
        domain_name=f'{args.tar}',
        dataset_path='/home/xingxuanzhang/yuanxue/data/DomainNet',
        batch_size=128,
        num_workers=16,
    ).get_dloader()

    results = infer_grid(w1, w2, w3, base_model, trainloader, testloader, G=G, margin=margin, update_bn=args.update_bn)

    if args.src3 == '':
        torch.save(results, f'src_{args.src1+args.src2}_tar_{args.tar}_weight_{args.weight_src1, 1-args.weight_src1}_G{G}_margin{margin}_NoupdateBN.pth')
    else:
        torch.save(results, f'src_{args.src1+args.src2+args.src3}_tar_{args.tar}_G{G}_margin{margin}.pth')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--src1', default='', help='source domain 1')
    parser.add_argument('--src2', default='', help='source domain 2')
    parser.add_argument('--src3', default='', help='source domain 3')
    parser.add_argument('--tar', default='', help='target domain')
    parser.add_argument('--weight_src1', type=float, help='weight of source1 in model fusion')
    parser.add_argument('--update_bn', action='store_true', help='whether to update BN statistic params')
    args = parser.parse_args()

    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    run(args, G=26, margin=1.5)
