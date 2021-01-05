import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import os
import time
import math
import argparse
import warnings
import pickle
import numpy as np

from functools import partial
from torch.utils.tensorboard import SummaryWriter
from monitors.metrics import write_metrics

import utils
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
from models.cifar100_presnet import preact_resnet32_cifar



# from quantizer.uniq import UniQQuantizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_network(dataset, arch):
    print('==> Building model..')
    if dataset == "imagenet":
        models = {
            'presnet18': PreActResNet18,
            'glouncv-alexnet': alexnet,
            'glouncv-presnet34': preresnet34,
            'glouncv-mobilenetv2_w1': mobilenetv2_w1
        }
        net = models.get(arch, None)()

    elif dataset == "cifar100":
        assert arch == "presnet32"
        net = preact_resnet32_cifar(num_classes=100)
    return net


def tweak_network(net, bit, arch, train_conf, quant_mode):
    train_mode, train_scheme = train_conf.split(".")
    assert bit > 1

    if train_mode.startswith("quan"):
        if train_scheme == "standard_uniq":
            from quantizer.standard_uniq import UniQConv2d, UniQInputConv2d, UniQLinear
            input_conv_layer = UniQInputConv2d
            conv_layer = UniQConv2d
            linear_layer = UniQLinear


        replacement_dict = {
            nn.Conv2d: partial(conv_layer, bit=bit, quant_mode=quant_mode),
            nn.Linear: partial(linear_layer, bit=bit, quant_mode=quant_mode)
        }
        exception_dict = {
            '__first__': partial(input_conv_layer, bit=8),
            '__last__': partial(linear_layer, bit=8),
        }

        if arch == "glouncv-mobilenetv2_w1":
            exception_dict['__last__'] = partial(conv_layer, bit=8)
        net = utils.replace_module(net,
                                   replacement_dict=replacement_dict,
                                   exception_dict=exception_dict,
                                   arch=arch)
    return net


def load_checkpoint(net, init_from):
    # Loading checkpoint
    # -----------------------------
    init_from = os.path.expanduser(init_from)
    if init_from and os.path.isfile(init_from):
        print('==> Initializing from checkpoint: ', init_from)
        checkpoint = torch.load(init_from)
        loaded_params = {}
        for k, v in checkpoint['net'].items():
            if not k.startswith("module."):
                loaded_params["module." + k] = v
            else:
                loaded_params[k] = v

        net_state_dict = net.state_dict()
        net_state_dict.update(loaded_params)
        net.load_state_dict(net_state_dict)
    else:
        warnings.warn("No checkpoint file is provided !!!")


def train(net, optimizer, trainloader, criterion, epoch, print_freq=10, cfg=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % print_freq == 0:
            print ("[Train] Epoch=", epoch,  " BatchID=", batch_idx, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'  \
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return (train_loss / batch_idx, correct / total)


def test(net, testloader, criterion, epoch, print_freq=10):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % print_freq == 0:
                print ("[Test] Epoch=", epoch, " BatchID=", batch_idx, 'Loss: %.3f | Acc: %.3f%% (%d/%d)' \
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100. * correct / total
    return (test_loss / batch_idx, correct / total, acc)


def simple_initialization(net, trainloader, num_batches=100,train_conf=None):
    net.train()
    from quantizer.standard_uniq import STATUS, UniQConv2d, UniQInputConv2d, UniQLinear
    for n, m in net.named_modules():
        if isinstance(m, UniQConv2d) or isinstance(
                m, UniQInputConv2d) or isinstance(m, UniQLinear):
            assert getattr(m, 'quan_a', None) != None
            assert getattr(m, 'quan_w', None) != None
            m.quan_a.set_init_state(STATUS.INIT_READY)
            m.quan_w.set_init_state(STATUS.INIT_READY)

    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(device)
        output = net(inputs)
        if batch_idx + 1 == num_batches: break

    for n, m in net.named_modules():
        if isinstance(m, UniQConv2d) or isinstance(
                m, UniQInputConv2d) or isinstance(m, UniQLinear):
            assert getattr(m, 'quan_a', None) != None
            assert getattr(m, 'quan_w', None) != None
            m.quan_a.set_init_state(STATUS.INIT_DONE)
            m.quan_w.set_init_state(STATUS.INIT_DONE)
    print ("Init stepsize done ~")


def create_train_params(model, main_wd, delta_wd, skip_keys, verbose=False):
    normal_params, stepsize_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        added = False
        for skip_key in skip_keys:
            if skip_key in name:
                if verbose: print ("Add:  ", name, " to stepsize_params")
                stepsize_params.append(param)
                added = True
                break
        if not added:
            normal_params.append(param)
    return [{'params': stepsize_params, 'weight_decay': delta_wd }, 
             {'params': normal_params, 'weight_decay': main_wd}]



@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print("Params: \n")
    print(OmegaConf.to_yaml(cfg))
    time.sleep(10)

    best_acc = 0
    start_epoch = 0
    working_dir = os.path.join(get_original_cwd(), cfg.output_dir,
                               cfg.train_id)
    os.makedirs(working_dir, exist_ok=True)
    writer = SummaryWriter(working_dir)

    # Setup data.
    # --------------------
    print('=> Preparing data..')
    trainloader, testloader = utils.get_dataloaders(
        dataset=cfg.dataset.name,
        batch_size=cfg.dataset.batch_size,
        data_root=cfg.dataset.data_root)

    net = setup_network(cfg.dataset.name, cfg.dataset.arch)
    net = tweak_network(net,
                        bit=cfg.quantizer.bit,
                        train_conf=cfg.train_conf,
                        quant_mode=cfg.quant_mode,
                        arch=cfg.dataset.arch)
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    print(net)
    print("Number of learnable parameters: ",
          sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6,
          "M")
    time.sleep(5)
    load_checkpoint(net, init_from=cfg.dataset.init_from)
    params = create_train_params(model=net, main_wd=cfg.quantizer.wd, delta_wd=0, skip_keys=['.delta', '.alpha'], verbose=cfg.verbose)
    criterion = nn.CrossEntropyLoss()

    # Setup optimizer
    # ----------------------------
    if cfg.quantizer.optimizer == 'sgd':
        print("=> Use SGD optimizer")
        optimizer = optim.SGD(params,
                              lr=cfg.quantizer.lr,
                              momentum=0.9,
                              weight_decay=cfg.quantizer.wd)

    elif cfg.quantizer.optimizer == 'adam':
        print("=> Use Adam optimizer")
        optimizer = optim.Adam(params,
                               lr=cfg.quantizer.lr,
                               weight_decay=cfg.quantizer.wd)


    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.dataset.epochs)

    if cfg.evaluate:
        print("==> Start evaluating ...")
        test(net, testloader, criterion, -1)
        exit()

    # -----------------------------------------------
    # Reset to 'warmup_lr' if we are using warmup strategy.
    if cfg.quantizer.enable_warmup:
        assert cfg.quantizer.bit == 1
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg.quantizer.warmup_lr

    # Initialization
    # ------------------------------------------------
    if cfg.quantizer.bit != 32 and "quan" in cfg.train_conf:
        simple_initialization(net,
                              trainloader,
                              num_batches=cfg.dataset.num_calibration_batches,
                              train_conf=cfg.train_conf)

    # Training
    # -----------------------------------------------
    save_checkpoint_epochs = list(range(10))

    for epoch in range(start_epoch, cfg.dataset.epochs):
        train_loss, train_acc1 = train(net, optimizer, trainloader, criterion, epoch, cfg=cfg)
        test_loss, test_acc1, curr_acc = test(net, testloader, criterion, epoch)

        # Save checkpoint.
        if curr_acc > best_acc:
            best_acc = curr_acc
            utils.save_checkpoint(net,
                                  lr_scheduler,
                                  optimizer,
                                  curr_acc,
                                  epoch,
                                  filename=os.path.join(
                                      working_dir, 'ckpt_best.pth'))
            print('Saving..')
            print('Best accuracy: ', best_acc)

        if lr_scheduler is not None:
            lr_scheduler.step()

        write_metrics(writer, epoch, net,  \
                    optimizer, train_loss, train_acc1, test_loss, test_acc1, prefix="Standard_Training")

    print('Best accuracy: ', best_acc)


if __name__ == "__main__":
    main()
