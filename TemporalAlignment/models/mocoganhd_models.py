# class that has additional functionalities provided in the mocogan_hd repository
"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""
import os

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def load_checkpoints(path, gpu):
    if gpu is None:
        ckpt = torch.load(path)
    else:
        loc = 'cuda:{}'.format(gpu)
        ckpt = torch.load(path, map_location=loc)
    return ckpt


def model_to_gpu(model, isTrain, gpu):
    if isTrain:
        if gpu is not None:
            model.cuda(gpu)
            model = DDP(model,
                        device_ids=[gpu],
                        find_unused_parameters=True)
        else:
            model.cuda()
            model = DDP(model, find_unused_parameters=True)
    else:
        model.cuda()
        model = nn.DataParallel(model)

    return model