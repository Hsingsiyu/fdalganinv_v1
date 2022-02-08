# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch.nn as nn
import torch.nn.functional as F
import torch
from .utils import ConjugateDualFunction

__all__ = ["fDALLoss"]


class fDALLoss(nn.Module):
    def __init__(self, divergence_name, gamma):
        super(fDALLoss, self).__init__()

        self.lhat = None
        self.phistar = None
        self.phistar_gf = None
        self.multiplier = 1.
        self.internal_stats = {}
        self.domain_discriminator_accuracy = -1

        self.gammaw = gamma
        self.phistar_gf = lambda t: ConjugateDualFunction(divergence_name).fstarT(t)
        self.gf = lambda v: ConjugateDualFunction(divergence_name).T(v)
        self.l_func=nn.MSELoss(reduction='none')  #TODO  reduction 'none' return a vector?

    def forward(self, y_s, y_t, y_s_adv, y_t_adv, K):
        # ---
        # y_s,y_t:hGE(x_s), hGE(x_t)
        # y_s_adv: h'(GE(x_s))
        # y_y_adv: h'(GE(x_t))

        v_s = y_s_adv
        v_t = y_t_adv


        l_s=self.l_func(v_s,y_s.detach())  #l(h'GE(x).hGE(x))   # why? 我这里不是对h‘进行了grl吗 为啥对y还要detach?
        l_t=self.l_func(v_t,y_t.detach())
        dst = self.gammaw * torch.mean(self.gf(l_s)) - torch.mean(self.phistar_gf(l_t))
        #dst = self.gammaw * torch.mean(l_s) - torch.mean(self.phistar_gf(l_t))


        self.internal_stats['lhatsrc'] = torch.mean(l_s).item()
        self.internal_stats['lhattrg'] = torch.mean(l_t).item()
        self.internal_stats['acc'] = self.domain_discriminator_accuracy
        self.internal_stats['dst'] = dst.item()

        # we need to negate since the obj is being minimized, so min -dst =max dst.
        # the gradient reversar layer will take care of the rest
        return -self.multiplier * dst
        # return self.multiplier * dst
