import torch
import torchvision

import numpy as np

class ANPWrapper:
    def __init__(self, model, tradeoff, lr, ep):
        '''
        creates an ANP system, an object that wraps around a pytorch model
        tradeoff: tradeoff coefficient, 'alpha' in paper
        lr: learning rate of the perturbation process
        ep: maximum perturbation/perturbation budget, epsilon
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

        self._save()
        self._generate_weight_mask_tensors()
        self._create_perturbation_tensors()

    def perturb_step(self, inputs, label):
        '''
        1 step of perturbation with dataset batch given
        '''
        # initialize perturbation values
        self._make_new_perturbation_values()

        # TODO: implement the ANP step method
        pass


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