import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50

from losses import ProportionLoss


def set_model_backbone(model_backbone='resnet18', is_pretrain=True):
    if model_backbone == 'resnet18':
        if is_pretrain:
            model = resnet18(weights='DEFAULT')
        else:
            model = resnet18()
    elif model_backbone == 'resnet34':
        if is_pretrain:
            model = resnet34(weights='DEFAULT')
        else:
            model = resnet34()
    elif model_backbone == 'resnet50':
        if is_pretrain:
            model = resnet50(weights='DEFAULT')
        else:
            model = resnet50()
    else:
        raise NameError('Unknown model backbone')

    return model

def logsumexp(x):  # to be exact, logmeanexp
    K = torch.tensor(x.size(1))
    x = -torch.log(K) + torch.logsumexp(x, dim=1)  # log-sum-exp trick
    return x

class model_LPLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.num_classes

        self.model = set_model_backbone(
            model_backbone=args.model_backbone, is_pretrain=args.is_pretrain)
        self.dim_feature = self.model.fc.in_features
        self.model.fc = nn.Sequential()

        self.aggregation_fn = args.aggregation_fn
        self.pnorm = args.pnorm

        self.classifier_pn = nn.Sequential(
            nn.Linear(self.dim_feature, 2),
        )

        self.instance_score = nn.Sequential(
            nn.Linear(self.dim_feature, 1),
        )

        self.classifier_p = nn.Sequential(
            nn.Linear(self.dim_feature, self.num_classes-1),
        )

        self.classifier_inst = nn.Sequential(
            nn.Linear(self.dim_feature, self.num_classes),
        )

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=6)

        self.attention = nn.Sequential(
            nn.Linear(self.dim_feature, self.dim_feature),  # V
            nn.Tanh(),
            nn.Linear(self.dim_feature, 1),  # w
        )
        self.tmp = args.tmp

        self.w_LLP, self.w_MIL = args.w_LLP, args.w_MIL
        self.w_n, self.w_p = args.w_n, args.w_p

        self.mil_weight = torch.tensor(
            [args.w_n, args.w_p]).float().to(args.device)
        self.CELoss = nn.CrossEntropyLoss(weight=self.mil_weight)
        self.PLoss = ProportionLoss(reduction='none')
        self.BCELoss = nn.BCELoss(reduction='none')
        self.device = args.device

    def feature_extractor(self, data):
        (_, _, c, w, h) = data.size()
        data = data.reshape(-1, c, w, h)
        f = self.model(data)
        return f

    def inst(self, data):
        (_, _, c, w, h) = data.size()
        data = data.reshape(-1, c, w, h)
        y = self.model(data)
        return y

    def _attn(self, x):
        A = self.attention(x)
        A = torch.transpose(A, 2, 1)
        A = A/self.tmp
        return A

    def _mil(self, A, x):
        f_mil = torch.matmul(A, x).squeeze(1) / A.sum(dim=-1)
        y_mil = self.classifier_pn(f_mil)
        return y_mil

    def _llp(self, A, x):
        y = self.classifier_p(x)
        confidence = F.softmax(y, dim=-1)
        pred_prop = torch.matmul(A, confidence).squeeze(1) / A.sum(dim=-1)
        return y, pred_prop

    def calculate_loss_posi(self, y_LLP, lp):
        input, target = y_LLP, lp
        is_posi = (target[:, 0] != 1).long()
        input[is_posi == 1, 0] = 0  # mask negative class
        input /= input.sum(dim=1, keepdims=True)  # normalize
        loss = is_posi * self.PLoss(input[:, 1:], target[:, 1:])
        return loss.mean()

    def calculate_loss_nega(self, y_LLP, lp):
        input, target = y_LLP, lp
        is_nega = (target[:, 0] == 1).long()
        loss = is_nega * self.PLoss(input, target)
        return loss.mean()

    def forward(self, data, lp):
        (b, n, _, _, _) = data.size()
        bag_label = (lp[:, 0] != 1).long()

        f = self.feature_extractor(data)
        logits_inst = self.instance_score(f)

        score_inst = F.sigmoid(logits_inst)
        A = score_inst.reshape(-1)

        score_inst = score_inst.reshape(b, n)

        ### MIL ###
        if self.aggregation_fn == 'mean':
            score_bag = score_inst.mean(-1)

        elif self.aggregation_fn == 'max':
            score_bag = torch.max(score_inst, dim=-1)[0]

        elif self.aggregation_fn == 'LSE':
            v = self.pnorm * score_inst
            v = logsumexp(v)
            v = torch.clamp(v, min=-0)
            score_bag = v / self.pnorm

        else:
            raise NameError('None MIL_f')

        y_inst = A.round()
        y_bag = score_bag.round()

        ### LLP ###
        A = A.reshape(b, 1, n)
        f = f.reshape(b, n, -1)
        logits, pred_prop = self._llp(A, f)  # LLP [B,N,C-1] [B,C-1]

        logits, A = logits.reshape(b*n, -1), A.squeeze(1).reshape(-1)
        y_inst_mil = A.round().reshape(-1)
        y_inst = logits.argmax(-1)+1
        y_inst[y_inst_mil == 0] = 0

        # loss
        loss_mil = self.BCELoss(score_bag, bag_label.float())
        loss_neg = self.mil_weight[0] * (bag_label == 0) * loss_mil
        loss_pos = self.mil_weight[1] * (bag_label == 1) * loss_mil
        loss_mil = (loss_neg + loss_pos).mean()

        loss_llp = self.PLoss(pred_prop, lp[:, 1:]).mean()

        loss = self.w_LLP * loss_llp + self.w_MIL * loss_mil

        return {'loss': loss, 'loss_mil': loss_mil, 'loss_llp': loss_llp, 'loss_neg': torch.zeros(1), 'logits_bag': score_bag, 'y_inst': y_inst, 'y_bag': y_bag, 'A': A, 'f_inst': f.reshape(b*n, -1), 'pred_prop': pred_prop}