#----------------------Training Real----------------------#
# single GPU training
out_file_name="../experiments/train_InvDN.out"
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/train_InvDN.yml > $out_file_name 2>&1 &