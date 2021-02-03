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
from monitors.common import SingleBatchStatisticsPrinter


import utils
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
from models.cifar100_presnet import preact_resnet32_cifar, preact_resnet20_cifar
from models.cifar100_presnet_standard import preact_resnet32_cifar as preact_resnet32_cifar_standard
from models.cifar100_presnet_standard import preact_resnet20_cifar as preact_resnet20_cifar_standard


import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# from quantizer.uniq import UniQQuantizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_network(dataset, arch, num_classes=10):
    print('==> Building model..')
    if dataset == "imagenet":
        models = {
            'presnet18': PreActResNet18,
            'glouncv-alexnet': alexnet,
            'glouncv-presnet34': preresnet34,
            'glouncv-mobilenetv2_w1': mobilenetv2_w1
        }
        net = models.get(arch, None)()

    elif dataset == "cifar100" or dataset == "cifar10":
        if arch == "presnet32":
            net = preact_resnet32_cifar(num_classes=num_classes)
        elif arch == "presnet32-standard":
            net = preact_resnet32_cifar_standard(num_classes=num_classes)


        elif arch == "presnet20":
            net = preact_resnet20_cifar(num_classes=num_classes)
        elif arch == "presnet20-standard":
            net = preact_resnet20_cifar_standard(num_classes=num_classes)


        else:
            raise ValueError("Unsupported")
    return net


def tweak_network(net, bit, arch, train_conf, quant_mode, cfg):
    train_mode, train_scheme = train_conf.split(".")
    assert bit > 1

    if train_mode.startswith("quan") or train_mode.startswith("mod"):
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


        if train_scheme == "lsq":
            from quantizer.lsq import Conv2dLSQ, InputConv2dLSQ, LinearLSQ
            input_conv_layer = InputConv2dLSQ
            conv_layer = Conv2dLSQ
            linear_layer = LinearLSQ
            replacement_dict = {
                nn.Conv2d: partial(conv_layer, bit=bit),
                nn.Linear: partial(linear_layer, bit=bit)
            }
            exception_dict = {
                '__first__': partial(input_conv_layer, bit=8),
                '__last__': partial(linear_layer, bit=8),
            }


        if train_scheme == "condconv":
            from quantizer.condconv import Dynamic_conv2d
            replacement_dict = { nn.Conv2d: partial(Dynamic_conv2d, K=cfg.K)}
            exception_dict = {}



        if train_scheme == "condlsqconv":
            from quantizer.condlsq import Dynamic_LSQConv2d
            replacement_dict = { nn.Conv2d: partial(Dynamic_LSQConv2d, K=cfg.K)}
            exception_dict = {}
            # exception_dict = { '__first__': nn.Conv2d,  '__last__': nn.Linear,}         



        # if arch == "glouncv-mobilenetv2_w1":
        #     exception_dict['__last__'] = partial(conv_layer, bit=8)
        net = utils.replace_module(net,
                                   replacement_dict=replacement_dict,
                                   exception_dict=exception_dict,
                                   arch=arch)

        if train_scheme == "condconv" or train_scheme == "condlsqconv":
            m = net.conv1
            net.conv1 = nn.Conv2d(in_channels=m.in_channels, 
                out_channels=m.out_channels, kernel_size=m.kernel_size, 
                stride=m.stride, padding=m.padding, dilation=m.dilation, 
                groups=m.groups, bias=(m.bias!=None))

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

kl_criterion = nn.KLDivLoss(reduction='batchmean')
def train(net, optimizer, trainloader, criterion, epoch, print_freq=10, cfg=None, _register_hook=False, monitors=None,logdata ={}, update_params=True, working_dir="/tmp"):
    print('\nEpoch: %d' % epoch)
    if update_params:
        net.train()
    else:
        net.eval()

    train_loss = 0
    correct = 0
    total = 0

    if hasattr(cfg, "regularization_dist"):
        # regularization_dist = list(cfg.regularization_dist)

        # regularization_w = cfg.regularization_w

        # if epoch <=100:
        #     regularization_dist = [0.1,0.15,0.75]

        #     w_linspace =np.linspace(0, cfg.regularization_w, 100 + 1)
        #     regularization_w =  w_linspace[epoch-0]

        # elif epoch >100 and epoch <= 200:
        #     regularization_dist = [0.1,0.75, 0.15]

        #     w_linspace =np.linspace(0, cfg.regularization_w, 100 + 1)
        #     regularization_w =  w_linspace[epoch-100]
        # elif epoch >200 and epoch <= 250:
        #     regularization_dist = [0.75, 0.15, 0.1]
        #     w_linspace =np.linspace(0, cfg.regularization_w, 50 + 1)
        #     regularization_w =  w_linspace[epoch-200]


        # elif epoch >250 and epoch <= 300:
        #     regularization_dist = [1.0, 0.0, 0.0]
        #     w_linspace =np.linspace(0, cfg.regularization_w, 50 + 1)
        #     regularization_w =  w_linspace[epoch-250]


        # elif epoch >300 and epoch <= 350:
        #     regularization_dist = [1.0, 0.0, 0.0]
        #     w_linspace =np.linspace(0, cfg.regularization_w, 50 + 1)
        #     regularization_w =  w_linspace[epoch-300]
        a_s = np.linspace(0,1,350, dtype=np.double) * 1.0
        a = a_s[epoch]

        bc = 1.0 - a
        c = bc * 1.0/3.0
        b = bc * 2.0/3.0
        delta = 1.0 - (a+b+c)
        c = c + delta

        assert float(a)+float(b)+float(c) == 1.00
        regularization_dist = [float(a), float(b), float(c)]


    else:
        #print ("Unexpected regularization --- Exit....")
        #exit()
        regularization_dist = list(np.array([1.0/cfg.K] * cfg.K, dtype=np.float32))

    if _register_hook:
        [m.start_epoch("train", epoch) for m in monitors]

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if _register_hook:
            [m.start_update("train", epoch, batch_idx) for m in monitors]

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, raw = net(inputs)

        kl_losses = []
        for d in raw:
            if d == None: continue 
            # a = torch.log_softmax(d, dim=1)
            #b = torch.softmax(torch.tensor([regularization_dist] * a.size(0), requires_grad=False), dim=1).to(a.device)
            b = torch.tensor([regularization_dist] * d.size(0), requires_grad=False).to(d.device)
            _klloss = F.kl_div(d.log(), b, None, None, 'sum')
            kl_losses.append(_klloss) 

        loss = criterion(outputs, targets) + cfg.regularization_w * torch.stack(kl_losses).mean()

        loss.backward()
        if update_params:
            optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if _register_hook:

            from quantizer.condlsq import Dynamic_LSQConv2d as quantizer_fn
            for n, m in  net.named_modules():
                if isinstance(m, quantizer_fn): #and m.bit!=8 and m.bit!=32:
                    with torch.no_grad():
                        attention = monitors[0].tensors["module#" + n].attention( monitors[0].tensors["actin#" + n])#.argmax(dim=1) + 2 
                        if type(attention) == tuple:
                            attention = attention[0].argmax(dim=1) + 2 
                        else:
                            attention = attention.argmax(dim=1) + 2 

                    if n not in logdata: logdata[n] = list(np.array(range(cfg.K)) + 2 ) #[2, 3, 4, 5]
                    logdata[n] = logdata[n] + list(attention.cpu().detach().numpy())


            [m.end_update("train", epoch, batch_idx) for m in monitors]

        if batch_idx % print_freq == 0:
            print ("[Train] Epoch=", epoch,  " BatchID=", batch_idx, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'  \
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    if _register_hook:
        print ("Doing something here ...")


        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(6, 6, hspace=0.6,wspace=0.1)
        idx = 0 
        for k, v in logdata.items():
            ax = fig.add_subplot(gs[idx])
            ax.hist(v)
            ax.set_xlabel(k)
            idx +=1

        # ax1 = plt.subplot(gs[0])
        # ax2 = plt.subplot(gs[1])
        # ax3 = plt.subplot(gs[2])
        # ax4 = plt.subplot(gs[3])    
        os.makedirs(os.path.join(working_dir, "attention_hist"), exist_ok=True)        
        plt.savefig(os.path.join(working_dir, "attention_hist",   "epoch_" + str(epoch) + ".png"))
        [m.end_epoch("train", epoch) for m in monitors]


    if hasattr(cfg, 'enable_condconv') and cfg.enable_condconv:
        net.module.update_temperature()
    return (train_loss / batch_idx, correct / total)


def test(net, testloader, criterion, epoch, print_freq=10):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            

            net_outs = net(inputs)
            if type(net_outs) in (list, tuple):
                assert len(net_outs) == 2
                outputs, raw  = net_outs
            else:
                outputs = net_outs



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

    net = setup_network(cfg.dataset.name, cfg.dataset.arch, cfg.dataset.num_classes)
    net = tweak_network(net,
                        bit=cfg.quantizer.bit,
                        train_conf=cfg.train_conf,
                        quant_mode=cfg.quant_mode,
                        arch=cfg.dataset.arch,
                        cfg=cfg)
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
        test_loss, test_acc1, curr_acc = test(net, testloader, criterion, -1)
        print ("test_loss=", test_loss)
        print ("test_acc1=", test_acc1)
        print ("curr_acc=", curr_acc)

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
    monitors = [
            SingleBatchStatisticsPrinter(module_types=None, mode="train", max_iterations=10, max_epoch=cfg.dataset.epochs, 
                save=False, working_dir=working_dir, 
                prefix="standard",
                num_samples_to_save=cfg.dataset.batch_size, num_features_to_save=1),
        ]

    _ = [m.set_network(net) for m in monitors]
    logdata = {}

    if "train_eval" in cfg and cfg.train_eval == True:
        train_loss, train_acc1 = train(net, optimizer, trainloader, criterion, -1, cfg=cfg, _register_hook=True, monitors=monitors, logdata=logdata, update_params=False)
        print ("Write logs....")


        exit()


    for epoch in range(start_epoch, cfg.dataset.epochs):
        _register_hook = getattr(cfg, "register_hook", False) and  \
                    (epoch % cfg.monitor_interval==0 or epoch in save_checkpoint_epochs or epoch == cfg.dataset.epochs - 1)
        logdata = {}
        train_loss, train_acc1 = train(net, optimizer, trainloader, criterion, epoch, cfg=cfg, _register_hook=_register_hook, monitors=monitors, logdata=logdata, working_dir=working_dir)
        test_loss, test_acc1, curr_acc = test(net, testloader, criterion, epoch)

        with open(os.path.join(working_dir, 'monitor_data.pkl'), 'wb') as f:
            pickle.dump(logdata, f)

        del logdata
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
