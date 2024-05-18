import argparse
from pathlib import Path


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--is_train', default=1, type=int)
    parser.add_argument('--is_eval', default=1, type=int)

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--model_backbone', default='resnet18', type=str)
    parser.add_argument('--is_pretrain', default=1, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--mini_batch', default=16, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--is_early_stopping', default=0, type=int)
    parser.add_argument('--patience', default=30, type=int)
    parser.add_argument('--output_path', default='result/debug/', type=str)

    parser.add_argument('--dataset_dir', default='dataset/', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10','svhn'])
    parser.add_argument('--is_partial', default=1, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--num_instances', default=32, type=int)
    parser.add_argument('--kFold', default=0, type=int)

    parser.add_argument('--w_n', default=1, type=float)
    parser.add_argument('--w_p', default=1, type=float)
    parser.add_argument('--w_MIL', default=0.01, type=float)
    parser.add_argument('--w_LLP', default=1, type=float)
    parser.add_argument('--aggregation_fn', default='mean', type=str,  choices=['mean','max','attention','LSE'])
    parser.add_argument('--pnorm', default=5, type=float)
    parser.add_argument('--consistency', default='none', type=str, choices=['none','vat'])
    parser.add_argument('--tmp', default=1, type=float)

    parser.add_argument('--output_path_ours',
                        default='result/debug/', type=str)

    args = parser.parse_args()

    args.dataset_dir = Path(args.dataset_dir)
    args.output_path = Path(args.output_path)
    args.output_path.mkdir(parents=True, exist_ok=True)

    return args
