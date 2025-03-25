import torch
import torchvision
from torch.nn import functional as F

import numpy as np

class ANPWrapper:
    def __init__(self, model, tradeoff, lr, ep, loss_fn=None):
        '''
        creates an ANP system, an object that wraps around a pytorch model
        tradeoff: tradeoff coefficient, 'alpha' in paper
        lr: learning rate of the perturbation process
        ep: maximum perturbation/perturbation budget, epsilon
        loss_fn: loss function, default to torch.nn.CrossEntropyLoss()
        '''
        # the model to diagnose and fix
        self.model = model

        self.alpha = tradeoff
        self.lr = lr
        self.ep = ep

        # the perturbation and masks used in the algorithm
        self.weight_perturbations = {}   # delta
        self.bias_perturbations = {}     # xi
        self.weight_masks = {}           # m

        # a dict of dicts for mapping each layer (by name) to its set of extra params (delta, xi, m)
        self.layer_extra_params = {}
        # populate it
        for name, layer in self.model.named_modules():
            self.layer_extra_params[name] = {}

        self.backup = {}

        # set loss function
        if loss_fn == None:
            loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn = loss_fn

        self._save()
        self._generate_weight_mask_tensors()
        self._create_perturbation_tensors()

        # flag to switch between perturbation mode or not
        # we need to accumulate i.e. add the gradient of weight masks tensors (m) over the two modes
        self.use_perturbation = True

        # hook handles, for removing them later
        self.hook_handles = []

        # add the overwriting forward hooks to the layers for perturbation
        self.not_perturbed_layers = {}
        self._add_hooks()

        # clean up the unused extra params tensors created but not linked to the model
        self._clear_unused_extra_parameters()

        pms = [self.weight_masks[k] for k in self.weight_masks]
        # optimizer for weight masks
        self.weight_masks_optimizer = torch.optim.SGD(pms, lr=self.lr, momentum=0.9)

    def perturb_step(self, inputs, label):
        '''
        1 step of perturbation with dataset batch given
        '''
        # print('perturbation phase')
        # initialize perturbation values
        self._make_new_perturbation_values()
        # set flags so we can get the gradient of loss w.r.t. perturbations
        self.use_perturbation = True
        self._set_perturbation_tensor_require_grad(True)
        
        # perform prediction and compute loss, then backprop
        pred = self.model(inputs)
        loss = self.loss_fn(pred,label)
        loss.backward()
        
        # maximize perturbation parameter for this batch step
        self._set_perturbation_tensor_require_grad(False)
        self._maximize_perturbations()
        # clear their grad
        self._clear_perturbation_tensor_grad()

        # debug print
        # self._show_perturbations_tensors_minmax()

        # print('weight mask phase (with perturbation)')
        weight_mask_loss = 0
        
        # zero grads here since they would have accumulated in the previous loss.backward() call
        self.weight_masks_optimizer.zero_grad()
        # compute (1 − alpha) * L((m + delta) * w, (1 + xi) * b)
        # here use_perturbation is still True
        pred = self.model(inputs)
        loss = (1-self.alpha) * self.loss_fn(pred,label)
        weight_mask_loss += loss.item()
        # and backward
        loss.backward()

        # print('weight mask phase (without perturbation)')
        # then compute alpha * L(m * w, b)
        # with perturbation not present
        self.use_perturbation = False
        pred = self.model(inputs)
        loss = self.alpha * self.loss_fn(pred,label)
        weight_mask_loss += loss.item()
        # and backward, while gradients are accumulated in m
        loss.backward()

        # debug print
        # self._show_masks_tensors_grad()
        
        # then we step the optimizer to change the weight mask tensors
        self.weight_masks_optimizer.step()
        self._clamp_weight_mask_tensors()

        # debug print
        # self._show_masks_tensors_minmax()

        # TODO: finish implementing the ANP step method
        
        return weight_mask_loss


    '''
    below 2 methods are copied from
    https://www.kaggle.com/code/itsuki9180/introducing-adversarial-weight-perturbation-awp
    ''' 
    def _save(self):
        '''
        save the existing params of the model,
        since we are only changing delta, xi, and m, we need to restore the params every step
        '''
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self):
        """
        Restore model parameter to original position
        Call after loss.backward(), before optimizer.step()
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])

    '''
    for initial setups
    '''
    def _generate_weight_mask_tensors(self):
        '''
        initialize weight mask tensors to all 1
        '''
        for name, param in self.model.named_parameters():
            if name.endswith('weight'):
                # self.weight_masks[name] = torch.ones_like(param)

                # create a mask for number of neurons (not individual weights!)
                t_dim = len(param.shape)
                n_dim = [param.shape[0],] + [1,] * (t_dim - 1)
                self.weight_masks[name] = torch.ones(n_dim, device=param.device)
                
                self.layer_extra_params[name.replace('.weight', '')]['m'] = self.weight_masks[name]
                self.weight_masks[name].requires_grad = True
                
    def _create_perturbation_tensors(self):
        '''
        create the tensors for perturbation
        '''
        for name, param in self.model.named_parameters():
            if name.endswith('weight'):
                # self.weight_perturbations[name] = torch.zeros_like(param)

                # create a perturbation parameter for number of neurons (not individual weights!)
                t_dim = len(param.shape)
                n_dim = [param.shape[0],] + [1,] * (t_dim - 1)
                self.weight_perturbations[name] = torch.zeros(n_dim, device=param.device)
                
                self.layer_extra_params[name.replace('.weight', '')]['delta'] = self.weight_perturbations[name]
            elif name.endswith('bias'):
                # self.bias_perturbations[name] = torch.zeros_like(param)

                # create a perturbation parameter for number of neurons (not individual weights!)
                t_dim = len(param.shape)
                n_dim = [param.shape[0],] + [1,] * (t_dim - 1)
                self.bias_perturbations[name] = torch.zeros(n_dim, device=param.device)
                
                self.layer_extra_params[name.replace('.bias', '')]['xi'] = self.bias_perturbations[name]

    def _make_new_perturbation_values(self):
        '''
        generate new values for the perturbation tensors
        used in beginning of each batch training step
        '''
        for name in self.weight_perturbations:
            self.weight_perturbations[name].uniform_(-self.ep, self.ep)
        for name in self.bias_perturbations:
            self.bias_perturbations[name].uniform_(-self.ep, self.ep)

    def _set_perturbation_tensor_require_grad(self, require_grad):
        '''
        set the require grad flag of the perturbation tensors
        used in beginning of each batch training step
        '''
        for name in self.weight_perturbations:
            self.weight_perturbations[name].requires_grad = require_grad
        for name in self.bias_perturbations:
            self.bias_perturbations[name].requires_grad = require_grad

    def _clear_perturbation_tensor_grad(self):
        '''
        clear gradients of the perturbation tensors
        used in each batch training step
        '''
        for name in self.weight_perturbations:
            self.weight_perturbations[name].grad = None
        for name in self.bias_perturbations:
            self.bias_perturbations[name].grad = None

    def _maximize_perturbations(self):
        '''
        maximizes the perturbation parameters
        in each batch training step, we need the maxed perturbation for
        optimizing pruning flag tensors within each batch
        '''
        for name in self.weight_perturbations:
            # some tensors are not linked to the model, we skip them
            if self.weight_perturbations[name].grad is None:
                print(f'extra parameter with no .grad but not removed found, name: {name}')
                continue
            
            self.weight_perturbations[name] += self.weight_perturbations[name].grad.detach().sign() * self.ep * 2
            self.weight_perturbations[name].clamp_(-self.ep, self.ep)
        for name in self.bias_perturbations:
            # some tensors are not linked to the model, we skip them
            if self.bias_perturbations[name].grad is None:
                print(f'extra parameter with no .grad but not removed found, name: {name}')
                continue
            
            self.bias_perturbations[name] += self.bias_perturbations[name].grad.detach().sign() * self.ep * 2
            self.bias_perturbations[name].clamp_(-self.ep, self.ep)

    def _clamp_weight_mask_tensors(self):
        '''
        clamp weight mask tensors (m) to between 0 and 1
        called after every optimizer step to change m
        '''
        with torch.no_grad():
            for name in self.weight_masks:
                self.weight_masks[name].clamp_(0.0, 1.0)
    
    def _generate_overwrite_hook(self, layer_type, **kwargs):
        '''
        generate forward hook functions to add to modules of a model
        they overwrites the output by computing the forward() call with the perturbation parameters applying to the weights
        '''
        if layer_type == torch.nn.modules.linear.Linear:
            def fc_forward_hook(module, i, o):
                # print('fc forward hook called, use_perturbation:', self.use_perturbation)
                # change the forward call
                if self.use_perturbation:
                    return F.linear(i[0], (kwargs['m'] + kwargs['delta']) * module.weight, (1 + kwargs['xi']) * module.bias)
                else:
                    return F.linear(i[0], (kwargs['m']) * module.weight, module.bias)
            return fc_forward_hook
        elif layer_type == torch.nn.modules.conv.Conv2d:
            if 'xi' in kwargs:
                def conv2d_forward_hook(module, i, o):
                    # print('conv2d forward hook called, use_perturbation:', self.use_perturbation)
                    
                    # change the forward call
                    if self.use_perturbation:
                        return module._conv_forward(i[0], (kwargs['m'] + kwargs['delta']) * module.weight, (1 + kwargs['xi']) * module.bias)
                    else:
                        return module._conv_forward(i[0], (kwargs['m']) * module.weight, module.bias)
            else:
                def conv2d_forward_hook(module, i, o):
                    # print('conv2d forward hook (no bias) called, use_perturbation:', self.use_perturbation)
                    
                    # change the forward call
                    if self.use_perturbation:
                        return module._conv_forward(i[0], (kwargs['m'] + kwargs['delta']) * module.weight, module.bias)
                    else:
                        return module._conv_forward(i[0], (kwargs['m']) * module.weight, module.bias)
            return conv2d_forward_hook
        elif layer_type == torch.nn.modules.batchnorm.BatchNorm2d:
            def bn2d_forward_hook(module, i, o):
                '''
                the entire BatchNorm2d.forward() method
                copied from:
                https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/batchnorm.py
                https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/batchnorm.py
                and modified
                '''

                # exponential_average_factor is set to self.momentum
                # (when it is available) only so that it gets updated
                # in ONNX graph when this node is exported to ONNX.
                if module.momentum is None:
                    exponential_average_factor = 0.0
                else:
                    exponential_average_factor = module.momentum

                if module.training and module.track_running_stats:
                    # TODO: if statement only here to tell the jit to skip emitting this when it is None
                    if module.num_batches_tracked is not None:  # type: ignore[has-type]
                        module.num_batches_tracked.add_(1)  # type: ignore[has-type]
                        if module.momentum is None:  # use cumulative moving average
                            exponential_average_factor = 1.0 / float(module.num_batches_tracked)
                        else:  # use exponential moving average
                            exponential_average_factor = module.momentum

                r"""
                Decide whether the mini-batch stats should be used for normalization rather than the buffers.
                Mini-batch stats are used in training mode, and in eval mode when buffers are None.
                """
                if module.training:
                    bn_training = True
                else:
                    bn_training = (module.running_mean is None) and (module.running_var is None)

                r"""
                Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
                passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
                used for normalization (i.e. in eval mode when buffers are not None).
                """
                if self.use_perturbation:
                    return F.batch_norm(
                        i[0],
                        # If buffers are not to be tracked, ensure that they won't be updated
                        module.running_mean if not module.training or module.track_running_stats else None,
                        module.running_var if not module.training or module.track_running_stats else None,
                        (kwargs['m'] + kwargs['delta']) * module.weight,
                        (1 + kwargs['xi']) * module.bias,
                        bn_training,
                        exponential_average_factor,
                        module.eps,
                    )
                else:
                    return F.batch_norm(
                        i[0],
                        # If buffers are not to be tracked, ensure that they won't be updated
                        module.running_mean if not module.training or module.track_running_stats else None,
                        module.running_var if not module.training or module.track_running_stats else None,
                        (kwargs['m']) * module.weight,
                        module.bias,
                        bn_training,
                        exponential_average_factor,
                        module.eps,
                    )
            return bn2d_forward_hook
        return None

    def _add_hooks(self):
        '''
        add the overwriting hooks to the modules (layers) of the model
        '''
        for name, layer in self.model.named_modules():
            # extra params registered for this layer
            extra_params = self.layer_extra_params[name]
            if len(extra_params) == 0:
                continue
            # generate forward hook function and add
            # if this layer is a layer type thatwould need to use one
            forward_hook = self._generate_overwrite_hook(type(layer), **extra_params)
            if forward_hook is not None:
                handle = layer.register_forward_hook(forward_hook)
                # keep track of hook handles to remove them later if needed
                self.hook_handles.append(handle)
            else:
                self.not_perturbed_layers[name] = True

    def _remove_hooks(self):
        '''
        remove all hooks
        used after the perturbation training has completed 
        and we want to consolidate which neurons to remove
        '''
        for handle in self.hook_handles:
            handle.remove()

    def _prune_neurons(self, threshold):
        '''
        after ANP training completed, we will prune neurons that needs to be pruned
        all elements in the tensors in self.weight_masks have value in [0, 1]
        we prune the neurons per a threshold (paper states they just used 0.2)
        '''
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name.endswith('weight'):
                    if name in self.weight_masks:
                        param.copy_(param * (self.weight_masks[name] >= threshold))
    
    def _clear_unused_extra_parameters(self):
        '''
        clean up some tensors that are created but not used
        '''
        remove_list = []
        for name in self.weight_masks:
            if name.replace('.weight', '').replace('.bias', '') in self.not_perturbed_layers:
                remove_list.append(name)
        for r in remove_list:
            self.weight_masks.pop(r)

        remove_list = []
        for name in self.weight_perturbations:
            if name.replace('.weight', '').replace('.bias', '') in self.not_perturbed_layers:
                remove_list.append(name)
        for r in remove_list:
            self.weight_perturbations.pop(r)

        remove_list = []
        for name in self.bias_perturbations:
            if name.replace('.weight', '').replace('.bias', '') in self.not_perturbed_layers:
                remove_list.append(name)
        for r in remove_list:
            self.bias_perturbations.pop(r)
        pass

    def _show_masks_tensors_grad(self):
        print('weight masks grad status:', {name: self.weight_masks[name].grad is None for name in self.weight_masks})

    def _show_masks_tensors_minmax(self):
        print('weight masks min max:', {name: (self.weight_masks[name].min().item(), self.weight_masks[name].max().item()) for name in self.weight_masks})

    def _show_perturbations_tensors_minmax(self):
        print('weight perturbations min max:', {name: (self.weight_perturbations[name].min().item(), self.weight_perturbations[name].max().item()) for name in self.weight_perturbations})