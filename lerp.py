import copy
import torch

def lerp_multi(param_ls: list, weights=None):
    for idx, ttt in enumerate(param_ls):
        if not isinstance(ttt, dict):
            param_ls[idx] = param_ls[idx].state_dict()
        param_ls[idx] = {key: param_ls[idx][key].cpu() for key in param_ls[idx]}

    if weights == None:
        weights = [1/len(param_ls) for _ in param_ls]

    target_net = dict()
    for k in param_ls[0]:
        # running_mean, running_var  requires_grad=False
        # 不分开写会报错
        if 'running' in k:  
            fs = torch.zeros(param_ls[0][k].size(), requires_grad=False)
            for idx, net in enumerate(param_ls):
                fs = fs + net[k].data * weights[idx]
        else:
            fs = torch.zeros(param_ls[0][k].size(), requires_grad=True)
            for idx, net in enumerate(param_ls):
                fs = fs + net[k] * weights[idx]
        target_net[k] = fs
    return target_net