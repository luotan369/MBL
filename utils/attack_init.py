import os
import torch
import numpy as np
from datetime import datetime

import attacks
import data.datasets as datasets
from utils.attack_count import AttackCountingFunction
from utils.load_models import load_generator, load_imagenet_model, load_cifar_model,load_mnist_model
from utils.buffer import ImageBuffer, AttackListBuffer
from utils.surrogate_trainer import TrainModelSurrogate


def seed_init():
    SEED = 1
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)


def data_init(args):
    if args.dataset_name == 'imagenet':
        valid_set = datasets.imagenet(args.dataset_root, mode="validation")
        args.x_size = (3, 224, 224)
        args.y_size = (3, 224, 224)
    elif args.dataset_name == 'cifar10':
        valid_set = datasets.cifar10(args.dataset_root, mode="validation")
        args.x_size = (3, 32, 32)
        args.y_size = (3, 32, 32)
    elif args.dataset_name == 'mnist':
        valid_set = datasets.mnist(args.dataset_root, mode="validation")
        args.x_size = (1, 32, 32)
        args.y_size = (1, 32, 32)
    else:
        raise NotImplementedError
    dataloader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, drop_last=False) 
    return dataloader


def model_init(args):
    if args.dataset_name == 'imagenet':
        load_model = load_imagenet_model
    elif args.dataset_name == 'cifar10':
        load_model = load_cifar_model
    elif args.dataset_name == 'mnist':
        load_model = load_mnist_model
    else:
        raise NotImplementedError

    # Target model
    T = load_model(args.target_model_name, defence_method=args.defence_method)
    T.eval()

    surrogates, surrogate_optims = [], []

    for surrogate_names in args.surrogate_model_names.split(','):
        surrogate_model, optim = load_model(surrogate_names, require_optim=True)
        # surrogate_model.train()
        surrogates.append(surrogate_model)
        surrogate_optims.append(optim)
    # Counting function
    F = AttackCountingFunction(args.max_query)
    # MCG Generator
    G = load_generator(args)
    return T, G, surrogates, surrogate_optims, F


def attacker_init(args):
    dataset = args.dataset_name
    max_query = args.max_query
    targeted = args.targeted
    class_num = 1000 if dataset == 'imagenet' else 10
    if args.linf == 0.0325:
        args.linf=8./255
    # linf = 0.05 if dataset == 'imagenet' else 8. / 255
    args.class_num = class_num
    # args.linf = linf
    linf=args.linf

    if args.attack_method == 'square':
        attacker = attacks.SquareAttack(dataset_name=dataset, max_query=max_query, targeted=targeted, class_num=class_num, linf=linf)
    elif args.attack_method == 'signhunter':
        attacker = attacks.SignHunter(dataset_name=dataset, max_query=max_query, targeted=targeted, class_num=class_num, linf=linf)
    elif args.attack_method == 'cgattack':
        attacker = attacks.CGAttack(dataset_name=args.dataset_name, max_query=max_query, targeted=args.targeted, class_num=class_num, linf=linf, popsize=20)
    elif args.attack_method == 'surfree':
        attacker = attacks.SurFree(linf=linf)
    elif args.attack_method == 'NES':
        attacker = attacks.NESAttack(dataset_name=dataset, max_query=max_query, targeted=targeted, class_num=class_num, linf=linf)
    else:
        raise NotImplementedError
    return attacker


def buffer_init(args):
    mini_batch_size = args.finetune_mini_batch_size
    attack_method = args.attack_method
    buffer_limit = args.buffer_limit  

    image_buffer = ImageBuffer(batch_size=mini_batch_size)  # minibatch_size=20
    clean_buffer = ImageBuffer(batch_size=mini_batch_size) 
    adv_buffer = AttackListBuffer(attack_method=attack_method, batch_size=mini_batch_size, uplimit=buffer_limit) 
    if not args.finetune_perturbation:
        adv_buffer = None   
    return image_buffer, clean_buffer, adv_buffer


def trainer_init(args):
    trainer = TrainModelSurrogate()
    return trainer


def log_init(args):
    if args.log_root is not None:
        log_path = args.log_root
    else:
        os.makedirs('./logs', exist_ok=True)
        targeted = 'T' if args.targeted else 'UT'
        timestamp = datetime.now().strftime('%m%d%H%M')
        log_path = f'./logs/{args.dataset_name}_{targeted}_{args.target_model_name}_{args.attack_method}_{timestamp}'
    return log_path
