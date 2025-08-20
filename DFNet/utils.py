#coding=utf-8
import os
import cv2
import numpy as np
import torch


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0],-1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0],-1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 +smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1-num/den
    return loss.mean()
def split_map(datapath):
    print(datapath)
    for name in os.listdir(datapath+'/GT/'):
        mask = cv2.imread(datapath+'/GT/'+name,0)
        mask[mask < 10] = 0
        body = cv2.blur(mask, ksize=(5,5))
        body = cv2.distanceTransform(body, distanceType=cv2.DIST_L2, maskSize=5)
        body = body**0.5

        tmp  = body[np.where(body>0)]
        if len(tmp)!=0:
            body[np.where(body>0)] = np.floor(tmp/np.max(tmp)*255)

        # if not os.path.exists(datapath+'/skeleton/'):
        #     os.makedirs(datapath+'/skeleton/')
        # cv2.imwrite(datapath+'/skeleton/'+name, body)
        #
        # if not os.path.exists(datapath+'/contour/'):
        #     os.makedirs(datapath+'/contour/')
        # cv2.imwrite(datapath+'/contour/'+name, mask-body)

        if not os.path.exists(datapath+'/body/'):
            os.makedirs(datapath+'/body/')
        cv2.imwrite(datapath+'/body/'+name, body)

        if not os.path.exists(datapath+'/detail/'):
            os.makedirs(datapath+'/detail/')
        cv2.imwrite(datapath+'/detail/'+name, mask-body)
if __name__=='__main__':
    split_map('/media/magus/nvme2t/Datasets/Unaligned_RGBT/weakly_aligned/VT5000-Train_unalign')
