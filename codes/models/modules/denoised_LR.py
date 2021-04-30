import sys
sys.path.append('./')
import numpy as np
import torch
import glob
import cv2
from skimage import img_as_float32 as img_as_float
from skimage import img_as_ubyte
import time
import os
from codes.models.modules.VDN import VDN as DN
from codes.data.util import imresize_np

def denoise(noisy_path, pretrained_path, save_path, scale=4, LR_path=None):
    use_gpu = True
    C = 3
    dep_U = 4

    # load the pretrained model
    print('Loading the Model')
    checkpoint = torch.load(pretrained_path)
    net = DN(C, dep_U=dep_U, wf=64)
    if use_gpu:
        net = torch.nn.DataParallel(net).cuda()
        net.load_state_dict(checkpoint)
    else:
        load_state_dict_cpu(net, checkpoint)
    net.eval()

    files = glob.glob(os.path.join(noisy_path, '*.png'))

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for i in range(len(files)):
        im_noisy = cv2.imread(files[i])[:, :, ::-1]
        im_noisy = img_as_float(cv2.cvtColor(im_noisy, cv2.COLOR_BGR2RGB))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1))[np.newaxis,])

        _, C, H, W = im_noisy.shape
        if H % 2**dep_U != 0:
            H -= H % 2**dep_U
        if W % 2**dep_U != 0:
            W -= W % 2**dep_U
        im_noisy = im_noisy[:H, :W, ]

        if use_gpu:
            im_noisy = im_noisy.cuda()
            print('Begin Testing on GPU')
        else:
            print('Begin Testing on CPU')
        with torch.autograd.set_grad_enabled(False):
            tic = time.time()
            phi_Z = net(im_noisy, 'test')
            toc = time.time() - tic
            err = phi_Z.cpu().numpy()
            print('Time: %.5f' % toc)
        if use_gpu:
            im_noisy = im_noisy.cpu().numpy()
        else:
            im_noisy = im_noisy.numpy()
        im_denoise = im_noisy - err[:, :C, ]
        im_denoise = np.transpose(im_denoise.squeeze(), (1, 2, 0))
        im_denoise = img_as_ubyte(im_denoise.clip(0, 1))
        file_name = files[i].split('/')[-1]
        cv2.imwrite(os.path.join(save_path, file_name), im_denoise)

        if not LR_path is None:
            if not os.path.exists(LR_path):
                os.mkdir(LR_path)
            LR_denoise = imresize_np(im_denoise, 1 / scale, True)
            cv2.imwrite(os.path.join(LR_path, file_name), LR_denoise)

def load_state_dict_cpu(net, state_dict0):
    state_dict1 = net.state_dict()
    for name, value in state_dict1.items():
        assert 'module.'+name in state_dict0
        state_dict1[name] = state_dict0['module.'+name]
    net.load_state_dict(state_dict1)

def main():
    # Validation
    noisy_path = ''
    save_path = ''
    LR_path = ''

    denoise(noisy_path, pretrained_path, save_path, 4, LR_path)

if __name__ == '__main__':
    main()