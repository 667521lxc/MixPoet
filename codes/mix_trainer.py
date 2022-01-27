# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-03-31 22:31:29
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2019 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/.
Github: https://github.com/THUNLP-AIPoet.
'''
import numpy as np
import random
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from layers import LossWrapper, ScheduledOptim, ExponentialDecay, LinearDecay
from config import device
from logger import MixAUSLogger
import utils


class MixTrainer(object):  #这个train函数是三个train函数里面最麻烦的，但结构和另外两个差不多

    def __init__(self, hps):
        self.hps = hps

    def run_validation(self, epoch, mixpoet, tool, lr):#后面调用，验证
        logger = MixAUSLogger('valid')
        logger.set_batch_num(tool.valid_batch_num)
        logger.set_log_path(self.hps.valid_log_path, "")
        logger.set_rate('learning_rate', lr)
        logger.set_rate('teach_ratio', mixpoet.get_teach_ratio())
        logger.set_rate('noise_weight', self.noise_decay_tool.get_rate())

        for step in range(0, tool.valid_batch_num):

            batch = tool.valid_batches[step] #这个batch里面什么形状？？

            batch_keys = batch[0].to(device)#这些是怎么存储在batch里面的？？
            batch_poems = batch[1].to(device)
            batch_dec_inps = [dec_inp.to(device) for dec_inp in batch[2]]
            batch_labels = batch[3].to(device)
            batch_label_mask = batch[4].to(device)
            batch_lengths = batch[5].to(device)

            rec_loss, cl_loss_w, cl_loss_xw, entro_loss, _, _, _ = \
                self.run_rec_step(mixpoet, None, batch_keys, batch_poems,
                    batch_dec_inps, batch_labels, batch_label_mask, batch_lengths, True)#调用后面的函数

            logger.add_rec_losses(rec_loss, cl_loss_w, cl_loss_xw, entro_loss)

            dis_loss = self.run_dis_step(mixpoet, None, batch_keys,
                batch_poems, batch_labels, batch_label_mask, True)#调用后面的函数

            logger.add_dis_loss(dis_loss)

            adv_loss = self.run_adv_step(mixpoet, None, batch_keys,
                batch_poems, batch_labels, batch_label_mask, True)#调用后面的函数

            logger.add_adv_loss(adv_loss) #上面是以类似的方式计算了三种loss，然后加入logger
            #这三个loss是什么？

        logger.print_log(epoch)

    #以下三个run开头的函数，都是训练的一步，在run train函数和run valid中被调用
    def run_adv_step(self, mixpoet, optimizer, keys, poems,
        labels, label_mask, valid=False): #这个函数的名字什么意思？？有关对应生成器
        if not valid:
            optimizer.zero_grad()

        z_prior, z_post, cb_label1, cb_label2 \
                = mixpoet.get_prior_and_posterior(keys, poems, labels, label_mask)

        noise_weight = self.noise_decay_tool.get_rate()

        eps1 = torch.randn_like(z_post) * noise_weight
        eps2 = torch.randn_like(z_prior) * noise_weight

        logits_real = mixpoet.layers['discriminator'](z_post+eps1, cb_label1, cb_label2)
        logits_fake = mixpoet.layers['discriminator'](z_prior+eps2, cb_label1, cb_label2)

        # RSGAN
        loss = F.logsigmoid(logits_fake-logits_real)
        loss = -loss.mean()

        if not valid:
            loss.backward()
            clip_grad_norm_(mixpoet.gen_parameters(), self.hps.clip_grad_norm)
            optimizer.step()

        return loss.item()#这个函数是在干什么？


    def run_dis_step(self, mixpoet, optimizer, keys, poems,
        labels, label_mask, valid=False):  #跟上面那个好像，是判别器
        if not valid:
            optimizer.zero_grad()

        with torch.no_grad():  #这是和上面不一样的地方
            z_prior, z_post, cb_label1, cb_label2 \
                = mixpoet.get_prior_and_posterior(keys, poems, labels, label_mask)

        noise_weight = self.noise_decay_tool.get_rate()

        eps1 = torch.randn_like(z_post) * noise_weight
        eps2 = torch.randn_like(z_prior) * noise_weight

        logits_real = mixpoet.layers['discriminator'](z_post+eps1, cb_label1, cb_label2)
        logits_fake = mixpoet.layers['discriminator'](z_prior+eps2, cb_label1, cb_label2)

        # RSGAN
        loss = F.logsigmoid(logits_real-logits_fake)
        loss = -loss.mean()

        if not valid:
            loss.backward()
            clip_grad_norm_(mixpoet.dis_parameters(), self.hps.clip_grad_norm)
            optimizer.step()

        return loss.item()


    def run_rec_step(self, mixpoet, optimizer, keys, poems, dec_inps,
        labels, label_mask, lengths, valid=False):#分类器 检索

        if not valid:
            optimizer.zero_grad()

        all_outs, combined_label1, combined_label2, \
            logits_cl_xw1, logits_cl_xw2, logits_cl_w1, logits_cl_w2, = \
            mixpoet(keys, poems, dec_inps, labels, label_mask, lengths)  #直接输入模型

        rec_loss = self.losswrapper.cross_entropy_loss(all_outs, dec_inps)

        cl_loss_w1, cl_loss_xw1, entro_loss_xw1 = self.losswrapper.cl_loss(logits_cl_w1,
            logits_cl_xw1, combined_label1, label_mask[:, 0])
        cl_loss_w2, cl_loss_xw2, entro_loss_xw2 = self.losswrapper.cl_loss(logits_cl_w2,
            logits_cl_xw2, combined_label2, label_mask[:, 1])

        cl_loss_w = cl_loss_w1 + cl_loss_w2
        cl_loss_xw = cl_loss_xw1 + cl_loss_xw2
        entro_loss_xw = entro_loss_xw1 + entro_loss_xw2

        loss = rec_loss + cl_loss_w + cl_loss_xw + entro_loss_xw

        if not valid:
            loss.backward()
            clip_grad_norm_(mixpoet.rec_parameters(), self.hps.clip_grad_norm)
            optimizer.step()

        return rec_loss.item(), cl_loss_w.item(), cl_loss_xw.item(),\
            entro_loss_xw.item(), all_outs, combined_label1, combined_label2

    # ----------------------------------------------------
    def gen_from_prior(self, mixpoet, keys, poems, dec_inps,
            labels, label_mask, lengths): #什么意思？

        with torch.no_grad():

            all_outs, _, _, _, _, _, _, = \
                mixpoet(keys, poems, dec_inps, labels, label_mask, lengths, True, 0.0)

        return all_outs

    # -------------------------------------------------------------------------
    def run_train(self, mixpoet, tool, optimizerRec, optimizerDis,
                optimizerGen, logger, epoch):

        logger.set_start_time()

        for step in range(0, tool.train_batch_num): #有多个batch

            batch = tool.train_batches[step]
            batch_keys = batch[0].to(device)
            batch_poems = batch[1].to(device)
            batch_dec_inps = [dec_inp.to(device) for dec_inp in batch[2]]
            batch_labels = batch[3].to(device)
            batch_label_mask = batch[4].to(device)
            batch_lengths = batch[5].to(device)

            #下面还是进行了三个训练，计算loss
            # train the classifier, recognition network and decoder 训练
            rec_loss, cl_loss_w, cl_loss_xw, entro_loss, outs_post, clabels1, clabels2 = \
                self.run_rec_step(mixpoet, optimizerRec,
                    batch_keys, batch_poems, batch_dec_inps,
                    batch_labels, batch_label_mask, batch_lengths)

            logger.add_rec_losses(rec_loss, cl_loss_w, cl_loss_xw, entro_loss)
            logger.set_rate("learning_rate", optimizerRec.rate())

            # train discriminator判别器     在GAN中，我们有Generator和Discriminator
            if logger.total_steps > self.hps.rec_warm_steps:
                dis_loss = 0
                for i in range(0, self.hps.ndis):  #这个ndis是什么？？？config.py没说
                    step_dis_loss = self.run_dis_step(mixpoet, optimizerDis,
                        batch_keys, batch_poems, batch_labels, batch_label_mask)#每次调用step函数训练
                    dis_loss += step_dis_loss
                dis_loss /= self.hps.ndis
                logger.add_dis_loss(dis_loss)
                logger.set_rate('noise_weight', self.noise_decay_tool.do_step())

            if logger.total_steps > self.hps.rec_warm_steps:
                # train prior and posterior generators生成器
                adv_loss = self.run_adv_step(mixpoet, optimizerGen,
                    batch_keys, batch_poems, batch_labels, batch_label_mask)

                logger.add_adv_loss(adv_loss)

            # temperature annealing温度退火 ，是模拟物理退火的过程而设计的随机优化算法，来优化目标函数
            mixpoet.set_tau(self.tau_decay_tool.do_step())
            logger.set_rate('temperature', self.tau_decay_tool.get_rate())

            if (step % 40 == 0) and (logger.total_steps > self.hps.rec_warm_steps):
                dist = utils.cal_distance(mixpoet, batch_keys, batch_poems,
                    batch_labels, batch_label_mask) #计算距离
                if not np.isnan(dist):
                    logger.add_distance(dist)
                #------------
                fadist = utils.factor_distance(mixpoet, batch_keys,
                    self.hps.n_class1, self.hps.n_class2, device)
                if not np.isnan(fadist):
                    logger.add_factor_distance(fadist)

            if step % self.hps.log_steps == 0: #打印日志
                logger.set_end_time()
                outs_prior = self.gen_from_prior(mixpoet,
                    batch_keys, batch_poems, batch_dec_inps,
                    batch_labels, batch_label_mask, batch_lengths)

                utils.sample_mix(batch_keys, batch_dec_inps, batch_labels,
                    clabels1, clabels2, outs_post, outs_prior, self.hps.sample_num, tool)
                logger.print_log()
                logger.draw_curves()
                logger.set_start_time()


    def train(self, mixpoet, tool): #这是一个比较完整的训练文件，里面了调用上面的一些函数
        #utils.print_parameter_list(mixpoet, "prior")
        # load data for pre-training 首先组建数据
        print ("building data for mixpoet...")
        tool.build_data(self.hps.train_data, self.hps.valid_data,
            self.hps.batch_size, mode='mixpoet_pre')

        print ("train batch num: %d" % (tool.train_batch_num))
        print ("valid batch num: %d" % (tool.valid_batch_num))

        # training logger
        logger = MixAUSLogger('train')
        logger.set_batch_num(tool.train_batch_num)
        logger.set_log_steps(self.hps.log_steps)
        logger.set_log_path(self.hps.train_log_path, self.hps.fig_log_path)
        logger.set_rate('learning_rate', 0.0)
        logger.set_rate('teach_ratio', 1.0)
        logger.set_rate('temperature', 1.0)

        # build optimizer 优化器，3个分别是谁的呢？？答：分别是分类器，鉴别器，生成器
        optRec = torch.optim.AdamW(mixpoet.rec_parameters(),
            lr=1e-3, betas=(0.9, 0.99), weight_decay=self.hps.weight_decay)
        optimizerRec = ScheduledOptim(optimizer=optRec, warmup_steps=self.hps.warmup_steps,
            max_lr=self.hps.max_lr, min_lr=self.hps.min_lr)

        optDis = torch.optim.AdamW(mixpoet.dis_parameters(),
            lr=1e-3, betas=(0.5, 0.99), weight_decay=self.hps.weight_decay)
        optimizerDis = ScheduledOptim(optimizer=optDis, warmup_steps=self.hps.warmup_steps,
            max_lr=self.hps.max_lr, min_lr=self.hps.min_lr, beta=0.5)

        optGen = torch.optim.AdamW(mixpoet.gen_parameters(),
            lr=1e-3, betas=(0.9, 0.99), weight_decay=self.hps.weight_decay)
        optimizerGen = ScheduledOptim(optimizer=optGen, warmup_steps=self.hps.warmup_steps,
            max_lr=self.hps.max_lr, min_lr=self.hps.min_lr, beta=0.55)

        # set loggers  这里下面的mixpoet是什么？？
        mixpoet.train()  

        #losswrapper：返回丢失函数的包装函数。这样做是为了在必要时启用loss函数的附加参数。
        self.losswrapper = LossWrapper(pad_idx=tool.get_PAD_ID(),
            sens_num=self.hps.sens_num, sen_len=self.hps.sen_len)  
        # change each epoch  学习率指数衰减
        tr_decay_tool = ExponentialDecay(self.hps.burn_down_tr, self.hps.decay_tr, self.hps.min_tr)
        # change each iteration
        self.tau_decay_tool = ExponentialDecay(0, self.hps.tau_annealing_steps, self.hps.min_tau)
        self.noise_decay_tool = LinearDecay(0, self.hps.noise_decay_steps, max_v=0.3, min_v=0.0)

        # -----------------------------------------------------------
        # train with all data  开始训练
        for epoch in range(1, self.hps.max_epoches+1): #这是epoch
            self.run_train(mixpoet, tool, optimizerRec, optimizerDis,
                optimizerGen, logger, epoch)  #调用函数训练

            if epoch % self.hps.validate_epoches == 0: #验证一下
                print("run validation...")
                mixpoet.eval()
                print ("in training mode: %d" % (mixpoet.training))
                self.run_validation(epoch, mixpoet, tool, optimizerRec.rate())
                mixpoet.train()
                print ("validation Done: %d" % (mixpoet.training))

            if (self.hps.save_epoches >= 0) and \
                (epoch % self.hps.save_epoches) == 0: #保存一下
                # save checkpoint
                print("saving model...")
                utils.save_checkpoint(self.hps.model_dir, epoch, mixpoet, "mixpre",
                    optimizerRec, optimizerDis, optimizerGen)

            logger.add_epoch()

            print ("teach forcing ratio decay...")
            mixpoet.set_teach_ratio(tr_decay_tool.do_step())    #这里有什么用呢？
            logger.set_rate('teach_ratio', tr_decay_tool.get_rate())

            print("shuffle data...")
            tool.shuffle_train_data()

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        # fine-tune with only labelled data 后面fine-tune，跟前面pre train步骤大致一样
        print ("building data for fine-tuning mixpoet...")
        tool.build_data(self.hps.train_data, self.hps.valid_data,
            self.hps.fbatch_size, mode='mixpoet_tune') #组建数据

        print ("train batch num: %d" % (tool.train_batch_num))
        print ("valid batch num: %d" % (tool.valid_batch_num))

        # training logger
        logger.set_batch_num(tool.train_batch_num)

        for epoch in range(1, self.hps.fmax_epoches+1):
            self.run_train(mixpoet, tool, optimizerRec, optimizerDis,
                optimizerGen, logger, epoch)

            if epoch % self.hps.validate_epoches == 0:
                print("run validation...")
                mixpoet.eval()
                print ("in training mode: %d" % (mixpoet.training))
                self.run_validation(epoch, mixpoet, tool, optimizerRec.rate())
                mixpoet.train()
                print ("validation Done: %d" % (mixpoet.training))

            if (self.hps.fsave_epoches >= 0) and \
                (epoch % self.hps.fsave_epoches) == 0:
                # save checkpoint
                print("saving model...")
                utils.save_checkpoint(self.hps.model_dir, epoch, mixpoet, "mixfine",
                    optimizerRec, optimizerDis, optimizerGen)

            logger.add_epoch()

            print ("teach forcing ratio decay...")
            mixpoet.set_teach_ratio(tr_decay_tool.do_step())
            logger.set_rate('teach_ratio', tr_decay_tool.get_rate())

            print("shuffle data...")
            tool.shuffle_train_data()
