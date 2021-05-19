import os
import cv2
import glob
import numpy as np

cropSize = 512
num = 100

# Crop SIDD
GT = glob.glob(os.path.join('/home/yang/Documents/Yang_Liu/Research_Projects/Data/SIDD/SIDD_Medium_Srgb/GT', '*.PNG'))
Noisy = glob.glob(os.path.join('/home/yang/Documents/Yang_Liu/Research_Projects/Data/SIDD/SIDD_Medium_Srgb/Noisy', '*.PNG'))

GT.sort()
Noisy.sort()

out_dir = "/media/yang/Seagate Expansion Drive/Yang_Liu/Data/SIDD/SIDD_Medium_Srgb_Patches_512"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(os.path.join(out_dir, 'GT')):
    os.mkdir(os.path.join(out_dir, 'GT'))
if not os.path.exists(os.path.join(out_dir, 'Noisy')):
    os.mkdir(os.path.join(out_dir, 'Noisy'))


for i in range(len(GT)):
    image = cv2.imread(GT[i])
    noisy_image = cv2.imread(Noisy[i])
    endw, endh = image.shape[0], image.shape[1]
    assert (endw >= cropSize) and (endh >= cropSize)
    for k in range(num):
        x = np.random.randint(0, endw - cropSize)
        y = np.random.randint(0, endh - cropSize)
        crop = image[x:(cropSize + x), y:(cropSize + y), :]
        noisy_crop = noisy_image[x:(cropSize + x), y:(cropSize + y), :]
        cv2.imwrite(os.path.join(out_dir, 'GT', '%d_%d.PNG'%(i, k)), crop)
        cv2.imwrite(os.path.join(out_dir, 'Noisy', '%d_%d.PNG' % (i, k)), noisy_crop)