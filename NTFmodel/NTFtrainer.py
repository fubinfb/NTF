"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
from itertools import combinations
from turtle import st
from click import style

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.trainer import cyclize, binarize_labels, expert_assign
from NTFbase_trainer import BaseTrainer
import base.utils as utils
from base.utils.visualize import save_tensor_as_mat
import os
from PIL import Image
import scipy.io as io
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



def to_batch(batch):
    in_batch = {
        "style_imgs": batch["style_imgs"].cuda(),
        "char_imgs": batch["char_imgs"].cuda(),
    }
    return in_batch


class NTFTrainer(BaseTrainer):
    def __init__(self, gen, disc, g_optim, d_optim, writer, logger, cfg, use_ddp):
        super().__init__(gen, disc, g_optim, d_optim, writer, logger, cfg, use_ddp)
        self.to_batch = to_batch

    def train(self, loader, max_step=100000):

        self.gen.train()
        if self.disc is not None:
            self.disc.train()

        # loss stats
        losses = utils.AverageMeters("g_total", "pixel", "Rec_pixel", "disc", "gen", "fm", "ac_s", "ac_c", "cross_ac_s", "cross_ac_c",
                                     "ac_gen_s", "ac_gen_c", "cross_ac_gen_s", "cross_ac_gen_c")
        # discriminator stats
        discs = utils.AverageMeters("real_font", "real_uni", "fake_font", "fake_uni",
                                    "real_font_acc", "real_uni_acc",
                                    "fake_font_acc", "fake_uni_acc")
        # etc stats
        stats = utils.AverageMeters("B", "ac_acc_s", "ac_acc_c", "ac_gen_acc_s", "ac_gen_acc_c")

        self.clear_losses()

        self.logger.info("Start training ...")

        for batch in cyclize(loader):
            epoch = self.step // len(loader)
            if self.use_ddp and (self.step % len(loader)) == 0:
                loader.sampler.set_epoch(epoch)

            style_imgs = batch["style_imgs"].cuda()
            char_imgs = batch["char_imgs"].cuda()

            trg_imgs = batch["trg_imgs"].cuda()
            trg_fids = batch["trg_fids"].cuda()
            trg_cids = batch["trg_cids"].cuda()

            B = len(trg_imgs)

            style_code1, char_code1 = self.gen.encode(style_imgs.flatten(0, 1), char_imgs.flatten(0, 1))

            final_img, img0 = self.gen.decode(style_code1, char_code1)

            stats.updates({
                "B": B,
            })


            real_font, real_uni, *real_feats = self.disc(
                trg_imgs, trg_fids, trg_cids, out_feats=self.cfg['fm_layers']
            )

            fake_font, fake_uni = self.disc(final_img.detach(), trg_fids, trg_cids)

            self.add_gan_d_loss([real_font, real_uni], [fake_font, fake_uni])

            self.d_optim.zero_grad()
            self.d_backward()
            self.d_optim.step()

            fake_font, fake_uni, *fake_feats = self.disc(
                final_img, trg_fids, trg_cids, out_feats=self.cfg['fm_layers']
            )
            self.add_gan_g_loss(fake_font, fake_uni)

            self.add_fm_loss(real_feats, fake_feats)

            def racc(x):
                return (x > 0.).float().mean().item()

            def facc(x):
                return (x < 0.).float().mean().item()

            discs.updates({
                "real_font": real_font.mean().item(),
                "real_uni": real_uni.mean().item(),
                "fake_font": fake_font.mean().item(),
                "fake_uni": fake_uni.mean().item(),

                'real_font_acc': racc(real_font),
                'real_uni_acc': racc(real_uni),
                'fake_font_acc': facc(fake_font),
                'fake_uni_acc': facc(fake_uni)
            }, B)

            self.add_pixel_loss(final_img, trg_imgs)

            self.add_Rec_pixel_loss(img0, char_imgs[:,0,:,:,:])

            self.g_optim.zero_grad()

            self.g_backward()
            self.g_optim.step()

            loss_dic = self.clear_losses()
            losses.updates(loss_dic, B)  # accum loss stats

            # EMA g
            self.accum_g()
            if self.is_bn_gen:
                self.sync_g_ema(batch)

            torch.cuda.synchronize()

            if self.cfg.rank == 0:
                if self.step % self.cfg.tb_freq == 0:
                    self.plot(losses, discs, stats)

                if self.step % self.cfg.print_freq == 0:
                    self.log(losses, discs, stats)
                    self.logger.debug("GPU Memory usage: max mem_alloc = %.1fM / %.1fM",
                                      torch.cuda.max_memory_allocated() / 1000 / 1000,
                                      torch.cuda.max_memory_cached() / 1000 / 1000)
                    losses.resets()
                    discs.resets()
                    stats.resets()


                if self.step > 0 and self.step % self.cfg.val_freq == 0:
                    epoch = self.step / len(loader)
                    self.logger.info("Validation at Epoch = {:.3f}".format(epoch))

                    self.save(self.cfg.save, self.cfg.get('save_freq', self.cfg.val_freq))
            else:
                pass

            if self.step >= max_step:
                break

            self.step += 1

        self.logger.info("Iteration finished.")

    def plot(self, losses, discs, stats):
        tag_scalar_dic = {
            'train/g_total_loss': losses.g_total.val,
            'train/pixel_loss': losses.pixel.val,
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

    def log(self, L, D, S):
        self.logger.info(
            f"Step {self.step:7d}\n"
            f"{'|D':<12} {L.disc.avg:7.3f} {'|G':<12} {L.gen.avg:7.3f} {'|FM':<12} {L.fm.avg:7.3f} {'|R_font':<12} {D.real_font_acc.avg:7.3f} {'|F_font':<12} {D.fake_font_acc.avg:7.3f} {'|R_uni':<12} {D.real_uni_acc.avg:7.3f} {'|F_uni':<12} {D.fake_uni_acc.avg:7.3f}\n"
            f"{'|AC_s':<12} {L.ac_s.avg:7.3f} {'|AC_c':<12} {L.ac_c.avg:7.3f} {'|cr_AC_s':<12} {L.cross_ac_s.avg:7.3f} {'|cr_AC_c':<12} {L.cross_ac_c.avg:7.3f} {'|AC_acc_s':<12} {S.ac_acc_s.avg:7.1%} {'|AC_acc_c':<12} {S.ac_acc_c.avg:7.1%}\n"
            f"{'|AC_g_s':<12} {L.ac_gen_s.avg:7.3f} {'|AC_g_c':<12} {L.ac_gen_c.avg:7.3f} {'|cr_AC_g_s':<12} {L.cross_ac_gen_s.avg:7.3f} {'|cr_AC_g_c':<12} {L.cross_ac_gen_c.avg:7.3f} {'|AC_g_acc_s':<12} {S.ac_gen_acc_s.avg:7.1%} {'|AC_g_acc_c':<12} {S.ac_gen_acc_c.avg:7.1%}\n"
            f"{'|L1':<12} {L.pixel.avg:7.3f} "
        )