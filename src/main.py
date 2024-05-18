import logging
import torch
import numpy as np
from time import time
from config import parse_option
from loader import set_loader
from losses import get_rampup_weight, set_consitency
from model import model_LPLP
from utils import *
from sklearn.metrics import confusion_matrix


def train_fn(model, loader, optimizer, epoch, history, args):
    s_time = time()

    model.train()
    consistency_criterion = set_consitency(args.consistency)

    loss_meter = AverageMeter()
    loss_mil_meter = AverageMeter()
    loss_llp_meter = AverageMeter()
    loss_neg_meter = AverageMeter()
    acc_bag_meter = AverageMeter()
    acc_inst_meter = AverageMeter()

    gt, pred = [], []

    pbar = set_pbar(loader)
    for _, batch in enumerate(loader):
        data, label, lp = batch[0], batch[1], batch[2]
        data, lp = data.to(args.device), lp.to(args.device)
        b = data.size(0)

        # consistency
        if args.consistency != 'none':
            consistency_loss = consistency_criterion(model, data)
            consistency_rampup = 0.4 * args.num_epochs * len(loader)
            alpha = get_rampup_weight(0.05, epoch, consistency_rampup)
            consistency_loss = alpha * consistency_loss
        else:
            consistency_loss = torch.tensor(0.)
        output = model(data, lp)
        output['loss'] += consistency_loss

        output['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()

        label = label.reshape(-1).cpu().detach().numpy()
        label_bag = (lp[:, 0] != 1).cpu().detach().numpy()
        y = output['y_inst'].cpu().detach().numpy()
        y_bag = output['y_bag'].cpu().detach().numpy()

        gt.extend(label)
        pred.extend(y)

        acc_bag_meter.update((label_bag == y_bag).mean(), b)
        acc_inst_meter.update((label == y).mean(), b)

        loss_meter.update(output['loss'].item(), b)
        loss_llp_meter.update(output['loss_llp'].item(), b)
        loss_mil_meter.update(output['loss_mil'].item(), b)

        pbar.set_description('loss mil:%.3f,llp:%.3f, acc inst:%.3f,bag:%.3f' %
                             (loss_mil_meter.avg, loss_llp_meter.avg, acc_inst_meter.avg, acc_bag_meter.avg))
        pbar.update()
    pbar.close()

    gt, pred = np.array(gt), np.array(pred)
    cm = confusion_matrix(y_true=gt, y_pred=pred)
    PC, mIoU = cal_PC(cm), cal_mIoU(cm)

    gt_binary, pred_binary = (gt != 0), (pred != 0)
    acc_pos = (gt_binary == pred_binary)[gt_binary == 1].mean()
    acc_neg = (gt_binary == pred_binary)[gt_binary == 0].mean()

    e_time = time()

    history['train_loss'].append(loss_meter.avg)
    history['train_loss_mil'].append(loss_mil_meter.avg)
    history['train_loss_llp'].append(loss_llp_meter.avg)
    history['train_loss_neg'].append(loss_neg_meter.avg)
    history['train_acc_bag'].append(acc_bag_meter.avg)
    history['train_acc_inst'].append(acc_inst_meter.avg)
    history['train_acc_pos'].append(acc_pos)
    history['train_acc_neg'].append(acc_neg)
    history['train_PC'].append(PC)
    history['train_mIoU'].append(mIoU)
    history['train_cm'].append(cm)

    logging.info('[%d/%d](%ds) train loss: %.3f, mil: %.3f, llp: %.3f, acc bag: %.3f, acc inst: %.3f, pos: %.3f, neg: %.3f, PC: %.3f, mIoU: %.3f' % (
        epoch+1, args.num_epochs, e_time-s_time,
        loss_meter.avg, loss_mil_meter.avg, loss_llp_meter.avg,
        acc_bag_meter.avg, acc_inst_meter.avg, acc_pos, acc_neg, PC, mIoU))

    return history


def val_fn(model, loader, history, args, is_test=0):
    s_time = time()

    model.eval()

    loss_meter = AverageMeter()
    loss_mil_meter = AverageMeter()
    loss_llp_meter = AverageMeter()
    loss_neg_meter = AverageMeter()
    acc_bag_meter = AverageMeter()
    acc_inst_meter = AverageMeter()

    gt, pred, A = [], [], []
    feature, gt_bag = [], []
    pred_prop = []

    pbar = set_pbar(loader)
    with torch.no_grad():
        for _, batch in enumerate(loader):
            data, label, lp = batch[0], batch[1], batch[2]
            data, lp = data.to(args.device), lp.to(args.device)
            (b, n, _, _, _) = data.size()

            output = model(data, lp)

            label = label.reshape(-1).cpu().detach().numpy()
            label_bag = (lp[:, 0] != 1).cpu().detach().numpy()
            y = output['y_inst'].cpu().detach().numpy()
            y_bag = output['y_bag'].cpu().detach().numpy()

            feature.extend(output['f_inst'].cpu().detach().numpy())
            gt_bag.extend(np.repeat(label_bag, n))

            gt.extend(label)
            pred.extend(y)
            A.extend(output['A'].cpu().detach().numpy())

            pred_prop.extend(np.eye(args.num_classes)[
                             output['y_inst'].cpu().detach().numpy().reshape(-1, n)].mean(-2))

            acc_bag_meter.update((label_bag == y_bag).mean(), b)
            acc_inst_meter.update((label == y).mean(), b)

            loss_meter.update(output['loss'].item(), b)
            loss_llp_meter.update(output['loss_llp'].item(), b)
            loss_mil_meter.update(output['loss_mil'].item(), b)

            pbar.set_description('loss mil:%.3f,llp:%.3f, acc inst:%.3f,bag:%.3f' %
                                 (loss_mil_meter.avg, loss_llp_meter.avg, acc_inst_meter.avg, acc_bag_meter.avg))
            pbar.update()
    pbar.close()

    gt, pred, A = np.array(gt), np.array(pred), np.array(A)
    cm = confusion_matrix(y_true=gt, y_pred=pred)
    PC, mIoU = cal_PC(cm), cal_mIoU(cm)

    gt_binary, pred_binary = (gt != 0), (pred != 0)
    acc_pos = (gt_binary == pred_binary)[gt_binary == 1].mean()
    acc_neg = (gt_binary == pred_binary)[gt_binary == 0].mean()
    cm_mil = confusion_matrix(y_true=gt_binary, y_pred=pred_binary)
    acc_mil = (gt_binary == pred_binary).mean()
    mIoU_mil = cal_mIoU(cm_mil)

    e_time = time()

    if is_test != 1:
        history['val_loss'].append(loss_meter.avg)
        history['val_loss_mil'].append(loss_mil_meter.avg)
        history['val_loss_llp'].append(loss_llp_meter.avg)
        history['val_loss_neg'].append(loss_neg_meter.avg)
        history['val_acc_bag'].append(acc_bag_meter.avg)
        history['val_acc_inst'].append(acc_inst_meter.avg)
        history['val_acc_pos'].append(acc_pos)
        history['val_acc_neg'].append(acc_neg)
        history['val_PC'].append(PC)
        history['val_mIoU'].append(mIoU)
        history['val_cm'].append(cm)

        logging.info('(%ds) val loss: %.3f, mil: %.3f, llp: %.3f, acc bag: %.3f, acc inst: %.3f, pos: %.3f, neg: %.3f, PC: %.3f, mIoU: %.3f' % (
            e_time-s_time, loss_meter.avg, loss_mil_meter.avg, loss_llp_meter.avg,
            acc_bag_meter.avg, acc_inst_meter.avg, acc_pos, acc_neg, PC, mIoU))

    else:
        history['test_loss'].append(loss_meter.avg)
        history['test_loss_mil'].append(loss_mil_meter.avg)
        history['test_loss_llp'].append(loss_llp_meter.avg)
        history['test_loss_neg'].append(loss_neg_meter.avg)
        history['test_acc_bag'].append(acc_bag_meter.avg)
        history['test_acc_inst'].append(acc_inst_meter.avg)
        history['test_acc_pos'].append(acc_pos)
        history['test_acc_neg'].append(acc_neg)
        history['test_PC'].append(PC)
        history['test_mIoU'].append(mIoU)
        history['test_cm'].append(cm)
        history['test_acc_mil'].append(acc_mil)
        history['test_mIoU_mil'].append(mIoU_mil)

        logging.info('(%ds) test loss: %.3f, mil: %.3f, llp: %.3f, acc bag: %.3f, acc inst: %.3f, pos: %.3f, neg: %.3f, PC: %.3f, mIoU: %.3f' % (
            e_time-s_time, loss_meter.avg, loss_mil_meter.avg, loss_llp_meter.avg,
            acc_bag_meter.avg, acc_inst_meter.avg, acc_pos, acc_neg, PC, mIoU))

    return history

def LPLP():
    args = parse_option()
    fix_seed(seed=args.seed)
    set_loger(args)

    ############ create loader ############
    train_loader, val_loader, test_loader = set_loader(
        args, is_full_proportion=False)
    model = model_LPLP(args).to(args.device)

    ############ training ############
    if args.is_train:
        history = {'best_loss': float('inf'), 'best_epoch': 0,
                   'train_loss': [], 'train_loss_mil': [], 'train_loss_llp': [], 'train_loss_neg': [], 'train_acc_bag': [], 'train_acc_inst': [], 'train_acc_pos': [], 'train_acc_neg': [], 'train_PC': [], 'train_mIoU': [], 'train_cm': [],
                   'val_loss': [], 'val_loss_mil': [], 'val_loss_llp': [], 'val_loss_neg': [], 'val_acc_bag': [], 'val_acc_inst': [], 'val_acc_pos': [], 'val_acc_neg': [], 'val_PC': [], 'val_mIoU': [], 'val_cm': [],
                   'test_loss': [], 'test_loss_mil': [], 'test_loss_llp': [], 'test_loss_neg': [], 'test_acc_bag': [], 'test_acc_inst': [], 'test_acc_pos': [], 'test_acc_neg': [], 'test_PC': [], 'test_mIoU': [], 'test_cm': []}
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        fix_seed(args.seed)
        for epoch in range(args.num_epochs):

            history = train_fn(model, train_loader,
                               optimizer, epoch, history, args)

            history = val_fn(model, val_loader, history, args)

            visualize_history(history, args.output_path)

            if history['best_loss'] > history['val_loss'][-1]:
                history['best_epoch'] = epoch
                history['best_loss'] = history['val_loss'][-1]
                save_model(model, optimizer, history,
                           args, epoch, args.output_path)

                if args.is_early_stopping:
                    early_stopping = 0
            else:
                if args.is_early_stopping:
                    early_stopping += 1
                    if early_stopping == args.patience:
                        break

    ############ testing ############
    if args.is_eval:
        state = torch.load(args.output_path/'model.pkl',
                           map_location=args.device)
        model.load_state_dict(state['model'])
        state['history']['test_acc_mil'] = []
        state['history']['test_mIoU_mil'] = []
        history = val_fn(model, test_loader, state['history'], args, is_test=1)
        torch.save(state, args.output_path/'model.pkl')
        plot_confusion_matrix(history['test_cm'][-1],
                              args.output_path/'cm_test.png',
                              'test confusion matrix\n Epoch %d: acc=%.3f, mIoU=%.3f' % (
                                  history['best_epoch']+1, history['test_acc_inst'][-1], history['test_mIoU'][-1]))


if __name__ == '__main__':
    LPLP()
