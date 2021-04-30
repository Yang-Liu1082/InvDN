import os
import scipy
import torch
import cv2
import scipy.io as sio
import h5py
from data.util import read_img_array
import logging
import argparse
import numpy as np
import options.options as option
import utils.util as util
from models import create_model

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

def SIDD_test(model, opt):
    dataset_dir = opt['name']

    out_dir = os.path.join('../experiments', dataset_dir)
    print(out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir = os.path.join(out_dir, 'SIDD_test')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # load info
    files = scipy.io.loadmat(os.path.join(opt['datasets']['test_1']['dataroot_Noisy'], 'BenchmarkNoisyBlocksSrgb.mat'))
    imgArray = files['BenchmarkNoisyBlocksSrgb']
    nImages = 40
    nBlocks = imgArray.shape[1]
    DenoisedBlocksSrgb = np.empty_like(imgArray)
    # process data
    for i in range(nImages):
        Inoisy = read_img_array(imgArray[i])
        Inoisy = torch.from_numpy(np.transpose(Inoisy, (0, 3, 1, 2))).type(torch.FloatTensor)

        for k in range(nBlocks):
            data = Inoisy[k].unsqueeze(dim=0)
            model.feed_test_data(data)
            if opt['self_ensemble']:
                model.test(opt['self_ensemble'])
            elif opt['mc_ensemble']:
                model.MC_test()
            else:
                model.test()

            img = model.fake_H.detach().float().cpu()
            Idenoised_crop = util.tensor2img_Real(img)  # uint8
            Idenoised_crop = np.transpose(Idenoised_crop, (1, 2, 0))
            DenoisedBlocksSrgb[i][k] = Idenoised_crop

            save_file = os.path.join(out_dir, '%d_%02d.PNG' % (i , k))
            cv2.imwrite(save_file, cv2.cvtColor(Idenoised_crop, cv2.COLOR_RGB2BGR))
            print('[%d/%d] is done\n' % (i+1, 40))

    save_file = os.path.join(out_dir, 'SubmitSrgb.mat') # SIDD_test_output
    sio.savemat(save_file, {'DenoisedBlocksSrgb': DenoisedBlocksSrgb, 'TimeMPSrgb' : 0.0})

def DND_test(model, opt):
    dataset_dir = opt['name']

    out_dir = os.path.join('../experiments', dataset_dir)
    print(out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir = os.path.join(out_dir, 'DND_test')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(os.path.join(out_dir, 'Submit')):
        os.mkdir(os.path.join(out_dir, 'Submit'))
    if not os.path.exists(os.path.join(out_dir, 'Images')):
        os.mkdir(os.path.join(out_dir, 'Images'))

    infos = h5py.File(os.path.join(opt['datasets']['test_2']['dataroot_Noisy'], 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    print('info loaded\n')
    # process data
    for i in range(50):
        filename = os.path.join(opt['datasets']['test_2']['dataroot_Noisy'], 'images_srgb', '%04d.mat'%(i+1))
        img = h5py.File(filename, 'r')
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)
        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T
        for k in range(20):
            idx = [int(boxes[k,0]-1),int(boxes[k,2]),int(boxes[k,1]-1),int(boxes[k,3])]
            Inoisy_crop = Inoisy[idx[0]:idx[1],idx[2]:idx[3],:].copy()
            Inoisy_crop = torch.from_numpy(np.transpose(Inoisy_crop, (2, 0, 1))).type(torch.FloatTensor)
            data = Inoisy_crop.unsqueeze(dim=0)
            model.feed_test_data(data)

            if opt['self_ensemble']:
                model.test(opt['self_ensemble'])
            elif opt['mc_ensemble']:
                model.MC_test()
            else:
                model.test()

            img = model.fake_H.detach().float().cpu()
            Idenoised_crop = util.tensor2img_Real(img, np.float32)  # uint8
            Idenoised_crop = np.transpose(Idenoised_crop, (1, 2, 0))
            # save denoised data
            save_file = os.path.join(out_dir, 'Submit', '%04d_%02d.mat'%(i+1,k+1))
            sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
            save_file = os.path.join(out_dir, 'Images', '%04d_%02d.PNG' % (i+1, k+1))
            cv2.imwrite(save_file, cv2.cvtColor(Idenoised_crop*255, cv2.COLOR_RGB2BGR))
            print('%s crop %d/%d' % (filename, k+1, 20))

        print('[%d/%d] %s done\n' % (i+1, 50, filename))


def DND_submissions_srgb(submission_folder):
    '''
    Bundles submission data for sRGB denoising

    submission_folder Folder where denoised images reside

    Output is written to <submission_folder>/bundled/. Please submit
    the content of this folder.
    '''
    out_folder = os.path.join(submission_folder, "bundled/")
    try:
        os.mkdir(out_folder)
    except:
        pass
    israw = False
    eval_version = "1.0"

    for i in range(50):
        Idenoised = np.zeros((20,), dtype=np.object)
        for bb in range(20):
            filename = '%04d_%02d.mat' % (i + 1, bb + 1)
            s = sio.loadmat(os.path.join(submission_folder, filename))
            Idenoised_crop = s["Idenoised_crop"]
            Idenoised[bb] = Idenoised_crop
        filename = '%04d.mat' % (i + 1)
        sio.savemat(os.path.join(out_folder, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )

def main():
    model = create_model(opt)
    SIDD_test(model, opt)
    DND_test(model, opt)
    submission_folder = os.path.join('../experiments', opt['name'], 'DND_test', 'Submit')
    DND_submissions_srgb(submission_folder)

if __name__ == "__main__":
    main()