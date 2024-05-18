import torch
import numpy as np
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def set_loger(args):
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(args.output_path/'training.log')
    logging.basicConfig(level=logging.INFO, handlers=[
                        stream_handler, file_handler])
    logging.info(args)


def set_pbar(loader):
    short_progress_bar = "{l_bar}{bar:5}{r_bar}{bar:-5b}"
    p_bar = tqdm(loader, leave=False, bar_format=short_progress_bar)
    return p_bar


def fix_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_PC(cm):
    num_classes = cm.shape[0]

    TP_c = np.zeros(num_classes)
    for i in range(num_classes):
        TP_c[i] = cm[i][i]

    FP_c = np.zeros(num_classes)
    for i in range(num_classes):
        FP_c[i] = cm[i, :].sum()-cm[i][i]

    PC = (TP_c/(TP_c+FP_c)).mean()

    return PC


def cal_mIoU(cm):
    num_classes = cm.shape[0]

    TP_c = np.zeros(num_classes)
    for i in range(num_classes):
        TP_c[i] = cm[i][i]

    FP_c = np.zeros(num_classes)
    for i in range(num_classes):
        FP_c[i] = cm[i, :].sum()-cm[i][i]

    FN_c = np.zeros(num_classes)
    for i in range(num_classes):
        FN_c[i] = cm[:, i].sum()-cm[i][i]

    mIoU = (TP_c/(TP_c+FP_c+FN_c)).mean()

    return mIoU


def save_model(model, optimizer, history, args, epoch, save_path):
    print('==> Saving...')
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'history': history,
        'args': args,
        'epoch': epoch,
    }
    torch.save(state, save_path/'model.pkl')
    del state

    plot_confusion_matrix(history['train_cm'][-1],
                          save_path/'cm_train.png',
                          'train confusion matrix\n Epoch %d: acc=%.3f, mIoU=%.3f' % (
        history['best_epoch']+1, history['train_acc_inst'][-1], history['train_mIoU'][-1]))
    plot_confusion_matrix(history['val_cm'][-1],
                          save_path/'cm_val.png',
                          'val confusion matrix\n Epoch %d: acc=%.3f, mIoU=%.3f' % (
        history['best_epoch']+1, history['val_acc_inst'][-1], history['val_mIoU'][-1]))


def plot_confusion_matrix(cm, path, title=''):
    cm = cm / cm.sum(axis=-1, keepdims=1)
    sns.heatmap(cm, annot=True, cmap='Blues_r', fmt='.2f')
    plt.xlabel('pred')
    plt.ylabel('GT')
    plt.title(title)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def visualize_history(history, path):
    plt.plot(history['train_acc_inst'],
             label='training inst. acc.', color='C0')
    plt.plot(history['val_acc_inst'],
             label='validation inst. acc.', color='C0', linestyle='dashed')
    plt.plot(history['train_acc_pos'],
             label='training acc.(pos.)', color='C1')
    plt.plot(history['val_acc_pos'], label='validation acc.(pos.)',
             color='C1', linestyle='dashed')
    plt.plot(history['train_acc_neg'], label='training acc.(neg.)',
             color='C2')
    plt.plot(history['val_acc_neg'], label='validation acc.(neg.)',
             color='C2', linestyle='dashed')
    plt.title('instance accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(path/'curve_acc_inst.png', dpi=300)
    plt.close()

    plt.plot(history['train_acc_bag'],
             label='training bag. acc.', color='C0')
    plt.plot(history['val_acc_bag'],
             label='validation bag. acc.', color='C0', linestyle='dashed')
    plt.title('bag accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(path/'curve_acc_bag.png', dpi=300)
    plt.close()

    plt.plot(history['train_mIoU'], label='training mIoU')
    plt.plot(history['val_mIoU'], label='validation mIoU')
    plt.title('mIoU')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(path/'curve_mIoU.png', dpi=300)
    plt.close()

    plt.plot(history['train_loss_llp'],
             label='training loss_llp', color='C0', alpha=0.5)
    plt.plot(history['val_loss_llp'], label='val loss_llp',
             color='C0', linestyle='dashed', alpha=0.5)
    plt.plot(history['train_loss_mil'],
             label='training loss_mil', color='C1', alpha=0.5)
    plt.plot(history['val_loss_mil'], label='val loss_mil',
             color='C1', linestyle='dashed', alpha=0.5)
    plt.plot(history['train_loss_neg'],
             label='training loss_neg', color='C2', alpha=0.5)
    plt.plot(history['val_loss_neg'], label='val loss_neg',
             color='C2', linestyle='dashed', alpha=0.5)
    plt.title('Loss')
    plt.legend()
    plt.savefig(path/'curve_loss.png', dpi=300)
    plt.close()
