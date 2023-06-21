import copy

import torch
import torch.nn.functional as F

import base.utils as utils
from base.trainer.trainer_utils import has_bn, load_checkpoint
from base.trainer.criterions import g_crit, d_crit, fm_crit
from pathlib import Path
import os
import scipy.io as io

class BaseTrainer:
    def __init__(self, gen, disc, g_optim, d_optim, writer, logger, cfg, use_ddp):
        self.step = 0

        self.gen = gen
        self.gen_ema = copy.deepcopy(self.gen)
        self.g_optim = g_optim

        self.is_bn_gen = has_bn(self.gen)
        self.disc = disc
        self.d_optim = d_optim

        self.writer = writer
        self.logger = logger

        self.cfg = cfg
        self.use_ddp = use_ddp

        self.set_ddp_models()
        if cfg.resume:
            self.step = load_checkpoint(cfg.resume, self.gen, self.disc, self.aux_clf, self.g_optim, self.d_optim, self.ac_optim, cfg.force_resume)
            logger.info("Resumed checkpoint from {} (Step {})".format(cfg.resume, self.step))

        self.g_losses = {}
        self.d_losses = {}
        self.frozen_ac_losses = {}

    def set_ddp_models(self):
        if self.use_ddp:
            self.logger.info("Setting DDP models...")
            self.gen = self.gen.module
            self.gen_ema = self.gen_ema.module
            self.disc = self.disc.module

    def clear_losses(self):
        """ Integrate & clear loss dict """
        # g losses
        loss_dic = {k: v.item() for k, v in self.g_losses.items()}
        loss_dic['g_total'] = sum(loss_dic.values())
        # d losses
        loss_dic.update({k: v.item() for k, v in self.d_losses.items()})

        self.g_losses = {}
        self.d_losses = {}
        self.frozen_ac_losses = {}

        return loss_dic

    def accum_g(self, decay=0.999):
        par1 = dict(self.gen_ema.named_parameters())
        par2 = dict(self.gen.named_parameters())
        
        for k in par1.keys():
            # print(k)
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=(1 - decay))

    def to_batch(self, batch):
        return

    def sync_g_ema(self, batch):
        org_train_mode = self.gen_ema.training # True
        with torch.no_grad():
            self.gen_ema.train()
            in_batch = self.to_batch(batch)
            self.gen_ema.infer(**in_batch)

        self.gen_ema.train(org_train_mode)

    def infer_loader(self, gen, loader):
        org_train_mode = gen.training
        gen.eval()
        with torch.no_grad():
            outs = []
            trgs = []
            for batch in loader:
                in_batch = self.to_batch(batch)
                out = gen.infer(**in_batch)

                outs.append(out.detach().cpu())
                trgs.append(batch["trg_imgs"])

            outs = torch.cat(outs).float()
            trgs = torch.cat(trgs)
        gen.train(org_train_mode)
        return outs, trgs
    
    def infer_save_img(self, loader, tag, n_row):
        grid_batches = self.infer_loader(self.gen_ema, loader)
        grid = utils.make_comparable_grid(*grid_batches[::-1], nrow=n_row)
        self.writer.add_image(tag, grid, global_step=self.step)

    def infer_loaderDF(self, gen, loader):
        save_root = "/home/fubin/FFG/check" 
        org_train_mode = gen.training
        gen.eval()
        with torch.no_grad():
            cout = 0
            for batch in loader:
                in_batch = self.to_batch(batch) # check the size of the batch bs,n_in,1,h,w ?
                out = gen.infer(**in_batch) # check the size of the out: bs,1,h,w?

                trgs = batch["trg_imgs"]
                N = len(trgs)
                for ii in range(N):
                    target_DF = trgs[ii,:,:,:]
                    pred_DF = out[ii,:,:,:]
                    
                    filename_GT = cout + "GT" + ".mat"
                    filename_Gen = cout + "Gen" + ".mat"
                    path_GT = os.path.join(save_root, filename_GT)
                    path_Gen = os.path.join(save_root, filename_Gen)
                    self.save_tensor_as_mat(target_DF,path_GT)
                    self.save_tensor_as_mat(pred_DF,path_Gen)
                    cout = cout + 1

        gen.train(org_train_mode)
        return out, trgs

    def train(self):
        return

    def add_loss(self, inputs, l_dict, l_key, weight, crit=F.l1_loss):
        loss = l_dict.get(l_key, 0.)
        loss += crit(*inputs) * weight
        l_dict[l_key] = loss

        return loss

    def add_pixel_loss(self, out, target):
        loss = self.add_loss(
            (out, target), self.g_losses, "pixel", self.cfg["pixel_w"], F.l1_loss
        )

        return loss

    def add_Rec_pixel_loss(self, out, target):
        loss = self.add_loss(
            (out, target), self.g_losses, "Rec_pixel", self.cfg["pixel_w"], F.l1_loss
        )

        return loss

    def add_gan_g_loss(self, *fakes):
        loss = self.add_loss(
            fakes, self.g_losses, "gen", self.cfg["gan_w"], g_crit
        )

        return loss

    def add_gan_d_loss(self, reals, fakes):
        loss = self.add_loss(
            (reals, fakes), self.d_losses, "disc", self.cfg["gan_w"], d_crit
        )

        return loss

    def add_fm_loss(self, real_feats, fake_feats):
        loss = self.add_loss(
            (real_feats, fake_feats), self.g_losses, "fm", self.cfg["fm_w"], fm_crit
        )

        return loss

    def d_backward(self):
        with utils.temporary_freeze(self.gen):
            d_loss = sum(self.d_losses.values())
            d_loss.backward()

    def g_backward(self):
        with utils.temporary_freeze(self.disc):
            g_loss = sum(self.g_losses.values())
            g_loss.backward()

    def save(self, method, save_freq=None):
        """
        Args:
            method: all / last
                all: save checkpoint by step
                last: save checkpoint to 'last.pth'
                all-last: save checkpoint by step per save_freq and
                          save checkpoint to 'last.pth' always
        """
        if method not in ['all', 'last', 'all-last']:
            return

        step_save = False
        last_save = False
        if method == 'all' or (method == 'all-last' and self.step % save_freq == 0):
            step_save = True
        if method == 'last' or method == 'all-last':
            last_save = True
        assert step_save or last_save

        save_dic = {
            'generator': self.gen.state_dict(),
            'generator_ema': self.gen_ema.state_dict(),
            'optimizer': self.g_optim.state_dict(),
            'step': int(self.step)
        }

        if self.disc is not None:
            save_dic['discriminator'] = self.disc.state_dict()
            save_dic['d_optimizer'] = self.d_optim.state_dict()

        ckpt_dir = self.cfg['work_dir'] / "checkpoints"
        step_ckpt_name = "{:06d}.pth".format(self.step)
        last_ckpt_name = "last.pth"
        step_ckpt_path = Path.cwd() / ckpt_dir / step_ckpt_name
        last_ckpt_path = ckpt_dir / last_ckpt_name

        log = ""
        if step_save:
            torch.save(save_dic, str(step_ckpt_path))
            log = "Checkpoint is saved to {}".format(step_ckpt_path)

        if last_save:
            utils.rm(last_ckpt_path)
            torch.save(save_dic, str(last_ckpt_path))
            log = "Checkpoint is saved to {}".format(last_ckpt_path)

        self.logger.info("{}\n".format(log))

        return str(last_ckpt_path)

    def plot(self, losses, discs, stats):
        tag_scalar_dic = {
            'train/g_total_loss': losses.g_total.val,
            'train/pixel_loss': losses.pixel.val
        }

        if self.disc is not None:
            tag_scalar_dic.update({
                'train/d_real_font': discs.real_font.val,
                'train/d_real_uni': discs.real_uni.val,
                'train/d_fake_font': discs.fake_font.val,
                'train/d_fake_uni': discs.fake_uni.val,

                'train/d_real_font_acc': discs.real_font_acc.val,
                'train/d_real_uni_acc': discs.real_uni_acc.val,
                'train/d_fake_font_acc': discs.fake_font_acc.val,
                'train/d_fake_uni_acc': discs.fake_uni_acc.val
            })

            if self.cfg['fm_w'] > 0.:
                tag_scalar_dic['train/feature_matching'] = losses.fm.val

        if self.aux_clf is not None:
            tag_scalar_dic.update({
                'train/ac_loss': losses.ac.val,
                'train/ac_acc': stats.ac_acc.val
            })

            if self.cfg['ac_gen_w'] > 0.:
                tag_scalar_dic.update({
                    'train/ac_gen_loss': losses.ac_gen.val,
                    'train/ac_gen_acc': stats.ac_gen_acc.val
                })

        self.writer.add_scalars(tag_scalar_dic, self.step)

    def plotCDF(self, losses, discs, stats):
        tag_scalar_dic = {
            'train/g_total_loss': losses.g_total.val,
            'train/pixel_loss': losses.pixel.val
        }

        if self.disc is not None:
            tag_scalar_dic.update({
                'train/d_real_font': discs.real_font.val,
                'train/d_real_uni': discs.real_uni.val,
                'train/d_fake_font': discs.fake_font.val,
                'train/d_fake_uni': discs.fake_uni.val,

                'train/d_real_font_acc': discs.real_font_acc.val,
                'train/d_real_uni_acc': discs.real_uni_acc.val,
                'train/d_fake_font_acc': discs.fake_font_acc.val,
                'train/d_fake_uni_acc': discs.fake_uni_acc.val
            })

            if self.cfg['fm_w'] > 0.:
                tag_scalar_dic['train/feature_matching'] = losses.fm.val

        self.writer.add_scalars(tag_scalar_dic, self.step)

    def log(self, losses, discs, stats):
        self.logger.info(
            "  Step {step:7d}: L1 {L.pixel.avg:7.4f}  D {L.disc.avg:7.3f}  G {L.gen.avg:7.3f}"
            "  FM {L.fm.avg:7.3f}  AC_loss {L.ac.avg:7.3f}  AC {S.ac_acc.avg:5.1%}  AC_gen {S.ac_gen_acc.avg:5.1%}"  # "  AC_fm {L.ac_fm.avg:7.3f}"
            "  R_font {D.real_font_acc.avg:7.3f}  F_font {D.fake_font_acc.avg:7.3f}"
            "  R_uni {D.real_uni_acc.avg:7.3f}  F_uni {D.fake_uni_acc.avg:7.3f}"
            "  B_stl {S.B_style.avg:5.1f}  B_trg {S.B_target.avg:5.1f}"
            .format(step=self.step, L=losses, D=discs, S=stats))