from __future__ import print_function

import shutil
import threading
from functools import partial

import numpy as np
import math
import sys
import os

import torch
import torch.nn as nn


class PropagationMonitor(object):

    # Note that update, or batch, starts from 0, whereas epoch starts from 1.
    def __init__(self, module_types, mode="train", max_iterations=-1, max_epoch=0):
        self.handles = []
        self.max_iterations = max_iterations
        self.max_epoch = max_epoch
        
        self.mode = mode
        self.update = 0
        self.epoch = -1
        if module_types is not None:
            self.module_types = {}
            for x in module_types:
                self.module_types[x] = None
        else:
            self.module_types = None
        
    def set_network(self, net):
        self.net = net

    def start_update(self, mode,  epoch, batch_idx):
        if self.mode!=mode:
            return

        if self.max_iterations==-1: # do it in start_epoch
            return
        if epoch>self.max_epoch and self.max_epoch!=-1:
            return

        if len(self.handles)==0 and batch_idx<=self.max_iterations:
            self.register_hooks()

        if batch_idx<=self.max_iterations:
            self.start_update_more(mode, epoch, batch_idx)

    def end_update(self, mode, epoch, batch_idx):
        if self.mode!=mode:
            return

        if self.max_iterations==-1: # do it in start_epoch
            return
        if epoch>self.max_epoch and self.max_epoch!=-1:
            return

        if len(self.handles)!=0 and batch_idx==self.max_iterations:
            self.unregister_hooks() # only one at the beginning and the end

        if batch_idx<=self.max_iterations:
            self.end_update_more(mode, epoch, batch_idx) # call in the region

    def start_epoch(self, mode, epoch):
        if self.mode!=mode:
            return

        self.epoch = epoch
        if len(self.handles)==0 and (self.max_epoch==-1 or
            epoch<=self.max_epoch) and self.max_iterations==-1:
            self.register_hooks()  # we postpone this to "update" functions

        if self.max_epoch==-1 or epoch<self.max_epoch:
            self.start_epoch_more(mode, epoch) # this is not postponed

    def end_epoch(self, mode, epoch):
        if self.mode!=mode:
            return

        # we unregister hooks at the end of every epoch
        # otherwise, we cannot save the model. 
        if len(self.handles)!=0 and self.max_iterations==-1:
            self.unregister_hooks()

        if self.max_epoch==-1 or epoch<=self.max_epoch:
            self.end_epoch_more(mode, epoch)


    def register_hooks(self):
        for n, m in self.net.named_modules():
            if m==self.net:
                continue
            if (self.module_types is None) or type(m) in self.module_types:
                m.name = n
                handle = m.register_forward_hook(
                            self.monitor_activations)
                self.handles.append(handle)
                handle = m.register_backward_hook(
                            self.monitor_gradients)
                self.handles.append(handle)

    def unregister_hooks(self):
        for h in self.handles:
            h.remove()
        del self.handles[:]
    

    def start_epoch_more(self, mode, epoch):
        pass

    def end_epoch_more(self, mode, epoch):
        pass

    def start_update_more(self, mode, epoch, batch_idx):
        pass

    def end_update_more(self, mode, epoch, batch_idx):
        pass

    # Note that due to DataParallel, this function can be called multiple times
    # for each module in an iteration
    def monitor_activations(self, module, input, output):
        pass

    # Note that due to DataParallel, this function can be called multiple times
    # for each module in an iteration
    def monitor_gradients(self, module, input, output):
        pass




class SingleBatchStatisticsPrinter(PropagationMonitor):

    def __init__(self, module_types, mode="train", prefix="", max_iterations= 3, max_epoch= 90,
        save=False, working_dir=".", num_samples_to_save=128, num_features_to_save=1):
        super(SingleBatchStatisticsPrinter, self).__init__(module_types, mode=mode,
                max_iterations=max_iterations, max_epoch=max_epoch)

        self.tensors = {}
        self.save = save
        self.prefix = prefix
        self.num_samples_to_save = num_samples_to_save
        self.num_features_to_save = num_features_to_save
        if save:
            self.log_dir = os.path.join(working_dir, "stat_logs")
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)
            # else:
            #     shutil.rmtree(self.log_dir)
            #     os.mkdir(self.log_dir)

        self.lock = threading.Lock()

    def start_epoch_more(self, mode, epoch):
        self.tensors = {}

    def end_epoch_more(self, mode, epoch):
        tensors_to_save = {}
        for k,v in self.tensors.items():
            if v == None: continue
            if k.startswith("module#"):
                tensors_to_save[k] = v
                continue
                
            if v.dim()==1: # this is just a wrapper module
                continue
            n = min(self.num_samples_to_save, v.shape[0])
            m = min(self.num_features_to_save, v.shape[1])
            tensors_to_save[k] = v[:n,:m].clone()

        if self.save: 
            path = os.path.join(self.log_dir, "epoch=%d#%s.%s.pth" % (epoch, self.prefix, mode))
            torch.save(tensors_to_save, path)

    def monitor_activations(self, module, input, output):
        if module.name=="":
            return

        #In a multi-gpu run, this is called repeatedly for each module.
        self.lock.acquire(True)
        if "actin#"+module.name in self.tensors: 
            self.lock.release()
            return
        if type(output)==tuple or type(output)==list:
            output = output[0]

        self.tensors["actin#"+module.name] = None
        # self.tensors["weight#"+module.name] = None
        # self.tensors["bias#"+module.name] = None

        self.lock.release()

        self.tensors["actin#"+module.name] = input[0].detach().clone()
        self.tensors["actout#"+module.name] = output.detach().clone()
        self.tensors["module#" + module.name] = module
        # self.tensors["weight#"+module.name] = getattr(module, "weight", None)
        # self.tensors["bias#"+module.name] = getattr(module, "bias", None)
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            if isinstance(module, nn.Linear):
                n = module.weight.data.shape[1]
            else:
                s = module.weight.data.shape
                n = s[1]*s[2]*s[3]
            var = input[0].data.var()
            e2 = ((input[0].data)**2).mean()
            #print('activations: %s (var:%f) (Ex^2:%f) (n:%d) ->  %f' %(module.name, var, e2, n, output.data[:,:].var()))
        else:
            var_in = input[0].data.var()
            var_out = output.data.var()
            gain = var_out/var_in
            #print('activations: %s %f  ->  %f (gain:%f)' %(module.name, var_in, var_out, gain))

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                for ch in range(1):
                    var_in = input[0].data[:,ch].var()
                    var_out = output.data[:,ch].var()
                    gain = var_out/var_in
                    #print('\t\t %s %f  ->  %f (gain:%f) (ch%d)' %(module.name, var_in, var_out, gain,ch))


    def monitor_gradients(self, module, inputa, output):
        assert len(output)==1
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            # p,p,g
            input_idx = 0 
        elif isinstance(module, nn.Linear):
            # b,g,w
            if module.bias is not None:
                input_idx = 1 
            else:
                input_idx = 0
        else: ## relu is 0
            # g
            input_idx = 0

        if module.name=="":
            return

        #In a multi-gpu run, this is called repeatedly for each module.
        self.lock.acquire(True)
        if "gradin#"+module.name in self.tensors:
            self.lock.release()
            return
        self.tensors["gradin#"+module.name] = None
        self.lock.release()

        self.tensors["gradin#"+module.name] = output[0].data.cpu()
        if inputa[input_idx] is not None:
            self.tensors["gradout#"+module.name] = inputa[input_idx].data.cpu()

        var_in = output[0].data.var()
        if inputa[input_idx] is None:
            pass
            #print('gradients: %s %f  ->  not computed' %(module.name, var_in))
        else:
            var_out = inputa[input_idx].data.var()
            gain = var_out/var_in
            
            #print('gradients: %s %f  ->  %f (gain: %f)' %(module.name, var_in, var_out, gain))

        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            for ch in range(1):
                var_in = output[0][:,ch].data.var()
                var_out = inputa[input_idx][:,ch].data.var()
                gain = var_out/var_in
                #print('\t\t %s %f  ->  %f (gain: %f) (ch%d)' %(module.name, var_in, var_out, gain, ch))
