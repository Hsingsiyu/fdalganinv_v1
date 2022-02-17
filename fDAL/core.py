# --coding:utf-8--
import torch.nn as nn
from .fDALLoss import fDALLoss
from .utils import WarmGRL
import torch
import copy

class fDALLearner(nn.Module):
    def __init__(self, backbone, taskhead, taskloss, divergence, batchsize,encoderdim,bootleneck=None, reg_coef=1, n_classes=-1,
                 aux_head=None,
                 grl_params=None,
                 Generator=None,
                 gpu_ids=None,
                 ):
        super(fDALLearner, self).__init__()
        self.backbone = backbone
        self.taskhead = taskhead
        self.taskloss = taskloss
        self.bootleneck = bootleneck
        self.n_classes = n_classes
        self.reg_coeff = reg_coef
        self.auxhead = aux_head if aux_head is not None else self.build_aux_head_()
        self.G=Generator
        self.fdal_divhead = fDALDivergenceHead(divergence, self.auxhead, n_classes=self.n_classes,
                                               grl_params=grl_params,
                                               reg_coef=reg_coef)
        self.batch_size=batchsize
        self.encode_dim=encoderdim
        self.code_loss=nn.L1Loss(reduction='mean')
        if gpu_ids is not None:
            assert len(gpu_ids) > 1
            self.auxhead= nn.DataParallel(self.auxhead, gpu_ids)
        ## todo fix param
        # for p in self.taskhead.parameters():
        #     p.requires_grad = False
        # for p in self.G.parameters():# 只是害怕！待确定
        #     p.requires_grad = False
        # for p in self.auxhead.parameters():
        #     p.requires_grad=True

    def build_aux_head_(self):
        # fDAL recommends the same architecture for both h, h'
        auxhead = copy.deepcopy(self.taskhead)
        #TODO 测试参数是否改变?
        auxhead.apply(lambda self_: self_.reset_parameters() if hasattr(self_, 'reset_parameters') else None)
        return auxhead

    def forward(self, x, y, src_size=-1, trg_size=-1):
        """
        :param x: tensor or tuple containing source and target input tensors.
        :param y: tensor or tuple containing source and target label tensors. (if unsupervised adaptation is a tensor with labels for source)
        :param src_size: src_size if specified. otherwise computed from input tensors
        :param trg_size: trg_size if specified. otherwise computed from input tensors

        :return: returns a tuple(tensor,dict). e.g. total_loss, {"pred_s": outputs_src, "pred_t": outputs_tgt, "taskloss": task_loss}

        """
        if isinstance(x, tuple):
            # assume x=x_source, x_target
            src_size = x[0].shape[0]
            trg_size = x[1].shape[0]
            #x = torch.cat((x[0], x[1]), dim=0)
            x_s=x[0]
            x_t=x[1]

        y_s = y
        y_t = None

        if isinstance(y, tuple):
            # assume y=y_source, y_target, otherwise assume y=y_source
            # warnings.warn_explicit('using target data')
            y_s = y[0]
            y_t = y[1]

        batch_size=x_s.shape[0]
        # latent   E(x)
        z_s=self.backbone(x_s).view(batch_size, *self.encode_dim)
        z_t=self.backbone(x_t).view(batch_size, *self.encode_dim)

        # reconstruct image  G(E(x))
        xrec_s=self.G(z_s)
        xrec_t=self.G(z_t)
        #TODO:save(x_recs,x_rect)

        #  hGE(x)
        outputs_src =self.taskhead(xrec_s)
        outputs_tgt =self.taskhead(xrec_t)


        source_label=self.taskhead(x_s) #h(x_S)

        # computing losses....
        # task loss in pixel
        task_loss_pix=self.taskloss(x_s,xrec_s) # L(x_s,G(E(x_s)))
        # task_loss_pix=0
        # task loss in code
        task_loss_z=self.code_loss(outputs_src,source_label) # L(h(x),hGE(x))
        # task loss in target if labels provided. Warning!. Only on semi-sup adaptation.
        task_loss=task_loss_pix+task_loss_z
        task_loss += 0.0 if y_t is None else self.taskloss(outputs_tgt, y_t)
        fdal_loss = 0.0
        if self.reg_coeff > 0.:
            # adaptation
            fdal_loss=self.fdal_divhead(xrec_s,xrec_t,outputs_src,outputs_tgt) # GE(x_s) , GE(x_t)  hGE(x_s) hGE(x_t)
            # together
            total_loss = task_loss + fdal_loss
        else:
            total_loss = task_loss
        #
        x_all = torch.cat([x_s, xrec_s, x_t, xrec_t], dim=0)
        h_all=torch.cat([source_label,outputs_src,outputs_tgt,self.fdal_divhead.internal_stats['hhat_s'],self.fdal_divhead.internal_stats['hhat_t']],dim=0)
        return total_loss, { "pix_loss": task_loss_pix,"code_loss":task_loss_z, "fdal_loss": fdal_loss/self.reg_coeff,
                            "fdal_src": self.fdal_divhead.internal_stats["lsrc"],
                            "fdal_trg": self.fdal_divhead.internal_stats["ltrg"],"x_all":x_all,"h_all":h_all}

    def get_reusable_model(self, pack=False):
        """
        Returns the usable parts of the model. For example backbone and taskhead. ignore the rest.

        :param pack: if set to True. will return a model that looks like taskhead( backbone(input)). Useful for inference.
        :return: nn.Module  or tuple of nn.Modules
        """
        if pack is True:
            return nn.Sequential(self.backbone, self.taskhead)
        return self.backbone, self.taskhead

class fDALDivergenceHead(nn.Module):
    def __init__(self, divergence_name, aux_head, n_classes, grl_params=None, reg_coef=1.):
        """
        :param divergence_name: divergence name (i.e pearson, jensen).
        :param aux_head: the auxiliary head refer to paper fig 1.
        :param n_classes:  if output is categorical then the number of classes. if <=1 will create a global discriminator.
        :param grl_params:  dict with grl_params.
        :param reg_coef: regularization coefficient. default 1.
        """
        super(fDALDivergenceHead, self).__init__()
        self.grl = WarmGRL(auto_step=True) if grl_params is None else WarmGRL(**grl_params)
        self.aux_head = aux_head
        self.fdal_loss = fDALLoss(divergence_name, gamma=1.0)
        self.internal_stats = self.fdal_loss.internal_stats
        self.n_classes = n_classes
        self.reg_coef = reg_coef


    def forward(self, features_s, features_t, pred_src, pred_trg) -> torch.Tensor:
        """
        :param features_s: features extracted by backbone on source data.
        :param features_t: features extracted by backbone on target data.
        :return: fdal loss
        """

        # h'(g(x)) auxiliary head output on source and target respectively.
        features_s=self.grl(features_s)
        features_t=self.grl(features_t)
        y_s_adv = self.aux_head(features_s) #  h'(GE(x_s))
        y_t_adv = self.aux_head(features_t) #  h'(GE(x_t))

        loss = self.fdal_loss(pred_src, pred_trg, y_s_adv, y_t_adv, self.n_classes)
        self.internal_stats = self.fdal_loss.internal_stats  # for debugging.

        return self.reg_coef * loss