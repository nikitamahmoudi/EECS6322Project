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

        # optimizer for weight masks
        self.weight_masks_optimizer = torch.optim.SGD(self.weight_masks,lr=self.lr)

    def perturb_step(self, inputs, label):
        '''
        1 step of perturbation with dataset batch given
        '''
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
        self._maximize_perturbations()
        # clear their grad
        self._clear_perturbation_tensor_grad()
        self._set_perturbation_tensor_require_grad(False)

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
        
        # then compute alpha * L(m * w, b)
        # with perturbation not present
        self.use_perturbation = False
        pred = self.model(inputs)
        loss = self.alpha * self.loss_fn(pred,label)
        weight_mask_loss += loss.item()
        # and backward, while gradients are accumulated in m
        loss.backward()

        # then we step the optimizer to change the weight mask tensors
        self.weight_masks_optimizer.step()

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
                self.weight_masks[name] = torch.ones_like(param)
                
    def _create_perturbation_tensors(self):
        '''
        create the tensors for perturbation
        '''
        for name, param in self.model.named_parameters():
            if name.endswith('weight'):
                self.weight_perturbations[name] = torch.zeros_like(param)
            elif name.endswith('bias'):
                self.bias_perturbations[name] = torch.zeros_like(param)

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
            self.weight_perturbations[name] += self.ep * self.weight_perturbations[name].grad.sign()
            self.weight_perturbations[name].clamp(-self.ep, self.ep)
        for name in self.bias_perturbations:
            self.bias_perturbations[name] += self.ep * self.bias_perturbations[name].grad.sign()
            self.bias_perturbations[name].clamp(-self.ep, self.ep)
    
    def _generate_overwrite_hook(self, layer_type, **kwargs):
        '''
        generate forward hook functions to add to modules of a model
        they overwrites the output by computing the forward() call with the perturbation parameters applying to the weights
        '''
        if layer_type == torch.nn.modules.linear.Linear:
            def fc_forward_hook(module, i, o):
                # change the forward call
                if self.use_perturbation:
                    return F.linear(i[0], (kwargs['m'] + kwargs['delta']) * module.weight, (1 + kwargs['xi']) * module.bias)
                else:
                    return F.linear(i[0], (kwargs['m']) * module.weight, module.bias)
            return fc_forward_hook
        elif layer_type == torch.nn.modules.conv.Conv2d:
            if 'xi' in kwargs:
                def conv2d_forward_hook(module, i, o):
                    # change the forward call
                    if self.use_perturbation:
                        return module._conv_forward(i[0], (kwargs['m'] + kwargs['delta']) * module.weight, (1 + kwargs['xi']) * module.bias)
                    else:
                        return module._conv_forward(i[0], (kwargs['m']) * module.weight, module.bias)
            else:
                def conv2d_forward_hook(module, i, o):
                    # change the forward call
                    if self.use_perturbation:
                        return module._conv_forward(i[0], (kwargs['m'] + kwargs['delta']) * module.weight, module.bias)
                    else:
                        return module._conv_forward(i[0], (kwargs['m']) * module.weight, module.bias)
            return conv2d_forward_hook
        elif layer_type == torch.nn.modules.batchnorm.BatchNorm2d:
            def bn2d_forward_hook(module, i, o):
                pass
            return None
        return None

    def _add_hooks(self):
        '''
        add the overwriting hooks to the modules (layers) of the model
        '''
        for name, layer in self.model.named_modules():
            forward_hook = self._generate_overwrite_hook(type(layer), )
            layer.register_forward_hook(forward_hook)