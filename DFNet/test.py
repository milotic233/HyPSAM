import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from swinv2_net import DFNet
from data import test_dataset
from options import opt
from collections import OrderedDict
import time
from os.path import splitext
# from ptflops.flops_counter import get_model_complexity_info


#set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


#load the model
model = DFNet()

base_weights = torch.load(opt.test_model)
new_state_dict = OrderedDict()
for k, v in base_weights.items():
    name = k[7:]
    new_state_dict[name] = v 
model.load_state_dict(new_state_dict)

print('Loading base network...')


model.cuda()
model.eval()

#test
test_data_root = opt.test_data_root
maps_path = opt.maps_path

test_sets = ['VT821','VT1000','VT5000/Test']

for dataset in test_sets:

    save_path = maps_path + dataset + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dataset_path = test_data_root + dataset
    test_loader = test_dataset(dataset_path, opt.testsize)
    total_time = 0
    frame_count = 0
    
    for i in range(test_loader.size):

        image, t, gt, (H, W), name = test_loader.load_data()
        image = image.cuda()
        t     = t.cuda()
        shape = (W,H)
        start_time = time.time()

        out_rgb, out_t, out_f, out_edge, out = model(image, t, shape)

        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time
        frame_count += 1


        res = out_f
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',save_path + name)
        cv2.imwrite(save_path + name,res*255)
    avg_time_per_frame = total_time / frame_count
    fps = 1.0 / avg_time_per_frame
    print(f'FPS: {fps:.2f}')

    print('Test Done!')