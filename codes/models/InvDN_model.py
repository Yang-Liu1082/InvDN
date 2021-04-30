import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss, Gradient_Loss, SSIM_Loss

logger = logging.getLogger('base')

class InvDN_Model(BaseModel):
    def __init__(self, opt):
        super(InvDN_Model, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])
            self.Rec_Forw_grad = Gradient_Loss()
            self.Rec_back_grad = Gradient_Loss()
            self.Rec_forw_SSIM = SSIM_Loss()
            self.Rec_back_SSIM = SSIM_Loss()

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data['GT'].to(self.device)  # GT
        self.noisy_H = data['Noisy'].to(self.device)  # Noisy

    def feed_test_data(self, data):
        self.noisy_H = data.to(self.device)  # Noisy

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def loss_forward(self, out, y):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
        # l_forw_grad = 0.1* self.train_opt['lambda_fit_forw'] * self.Rec_Forw_grad(out, y)
        # l_forw_SSIM = self.train_opt['lambda_fit_forw'] * self.Rec_forw_SSIM(out, y).mean()

        return l_forw_fit # + l_forw_grad + l_forw_SSIM

    def loss_backward(self, x, y):
        x_samples = self.netG(x=y, rev=True)
        x_samples_image = x_samples[:, :3, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples_image)
        l_grad_back_rec = 0.1*self.train_opt['lambda_rec_back'] * self.Rec_back_grad(x, x_samples_image)
        l_back_SSIM = self.train_opt['lambda_rec_back'] * self.Rec_back_SSIM(x, x_samples_image).mean()
        return l_back_rec + l_grad_back_rec + l_back_SSIM


    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        # forward
        self.output = self.netG(x=self.noisy_H)

        LR_ref = self.ref_L.detach()

        l_forw_ce = 0
        l_forw_fit = self.loss_forward(self.output[:, :3, :, :], LR_ref)

        # backward
        gaussian_scale = self.train_opt['gaussian_scale'] if self.train_opt['gaussian_scale'] != None else 1
        y_ = torch.cat((self.output[:, :3, :, :], gaussian_scale * self.gaussian_batch(self.output[:, 3:, :, :].shape)), dim=1)

        l_back_rec = self.loss_backward(self.real_H, y_)

        # total loss
        loss = l_forw_fit + l_back_rec + l_forw_ce
        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()

        # set log
        self.log_dict['l_forw_fit'] = l_forw_fit.item()
        self.log_dict['l_forw_ce'] = l_forw_ce
        self.log_dict['l_back_rec'] = l_back_rec.item()

    def test(self, self_ensemble=False):
        self.input = self.noisy_H

        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        self.netG.eval()
        with torch.no_grad():
            if self_ensemble:
                forward_function = self.netG.forward
                self.fake_H = self.forward_x8(self.input, forward_function, gaussian_scale)
            else:
                output = self.netG(x=self.input)
                self.forw_L = output[:, :3, :, :]
                y_forw = torch.cat((output[:, :3, :, :], gaussian_scale * self.gaussian_batch(output[:, 3:, :, :].shape)), dim=1)
                self.fake_H = self.netG(x=y_forw, rev=True)[:, :3, :, :]

        self.netG.train()

    def MC_test(self, sample_num=16, self_ensemble=False):
        self.input = self.noisy_H

        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        self.netG.eval()
        with torch.no_grad():
            if self_ensemble:
                forward_function = self.netG.forward
                self.fake_H = self.Multi_forward_x8(self.input, forward_function, gaussian_scale, sample_num)
            else:
                output = self.netG(x=self.input)
                self.forw_L = output[:, :3, :, :]
                fake_Hs = []
                for i in range(sample_num):
                    y_forw = torch.cat((output[:, :3, :, :], gaussian_scale * self.gaussian_batch(output[:, 3:, :, :].shape)), dim=1)
                    fake_Hs.append(self.netG(x=y_forw, rev=True)[:, :3, :, :])
                fake_H = torch.cat(fake_Hs, dim=0)
                self.fake_H = fake_H.mean(dim=0, keepdim=True)

        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['Denoised'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        out_dict['Noisy'] = self.noisy_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def forward_x8(self, x, forward_function, gaussian_scale):
        def _transform(v, op):
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            return ret

        noise_list = [x]
        for tf in 'v', 'h', 't':
            noise_list.extend([_transform(t, tf) for t in noise_list])

        lr_list = [forward_function(aug) for aug in noise_list]
        back_list = []
        for data in lr_list:
            y_forw = torch.cat((data[:, :3, :, :], gaussian_scale * self.gaussian_batch(data[:, 3:, :, :].shape)), dim=1)
            back_list.append(y_forw)
        sr_list = [forward_function(data, rev=True) for data in back_list]

        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output

    def Multi_forward_x8(self, x, forward_function, gaussian_scale, sample_num=16):
        def _transform(v, op):
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            return ret

        noise_list = [x]
        for tf in 'v', 'h', 't':
            noise_list.extend([_transform(t, tf) for t in noise_list])

        lr_list = [forward_function(aug) for aug in noise_list]
        sr_list = []
        for data in lr_list:
            fake_Hs = []
            for i in range(sample_num):
                y_forw = torch.cat((data[:, :3, :, :], gaussian_scale * self.gaussian_batch(data[:, 3:, :, :].shape)), dim=1)
                fake_Hs.append(self.netG(x=y_forw, rev=True)[:, :3, :, :])
            fake_H = torch.cat(fake_Hs, dim=0)
            fake_H = fake_H.mean(dim=0, keepdim=True)
            sr_list.append(fake_H)

        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output