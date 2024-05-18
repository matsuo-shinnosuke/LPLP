import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def set_consitency(consistency):
    if consistency == 'none':
        consistency_criterion = None
    elif consistency == 'vat':
        consistency_criterion = VATLoss()
    else:
        raise NameError('Unknown consistency criterion')
    
    return consistency_criterion

def cross_entropy_loss(input, target, eps=1e-8):
    # input = torch.clamp(input, eps, 1 - eps)
    loss = -target * torch.log(input+eps)
    return loss


class ProportionLoss(nn.Module):
    def __init__(self, metric="ce", consistency='none', eps=1e-8, reduction='mean'):
        super().__init__()
        self.metric = metric
        self.eps = eps
        self.reduction = reduction

        if consistency == 'none':
            self.consistency = None
        elif consistency == 'vat':
            self.consistency = VATLoss()
        else:
            raise NameError('Unknown consistency criterion')


    def forward(self, input, target):
        if self.metric == "ce":
            loss = cross_entropy_loss(input, target, eps=self.eps)
        elif self.metric == "l1":
            loss = F.l1_loss(input, target, reduction="none")
        elif self.metric == "mse":
            loss = F.mse_loss(input, target, reduction="none")
        else:
            raise NameError("metric {} is not supported".format(self.metric))

        loss = torch.mean(loss, dim=-1)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            loss = loss

        return loss


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def get_rampup_weight(weight, iteration, rampup):
    alpha = weight * sigmoid_rampup(iteration, rampup)
    return alpha


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


class VATLoss(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model.inst(x), dim=1)

        # prepare random unit tensor
        # d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = torch.randn_like(x)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model.inst(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model.inst(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds


class VATLoss_LPLP(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss_LPLP, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            # pred = F.softmax(model(x), dim=1)
            pred = model(x)[0]
            pred = pred.reshape(-1, pred.size(-1))

        # prepare random unit tensor
        # d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = torch.randn_like(x)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)[0]
                pred_hat = pred_hat.reshape(-1, pred_hat.size(-1))
                logp_hat = torch.log(pred_hat+1e-16)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')

                adv_distance.backward()
                if (torch.isnan(d.grad)).sum() > 1:
                    print('nan!')
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)[0]
            pred_hat = pred_hat.reshape(-1, pred_hat.size(-1))
            logp_hat = torch.log(pred_hat+1e-16)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        if torch.isnan(lds):
            print('nan!')

        return lds