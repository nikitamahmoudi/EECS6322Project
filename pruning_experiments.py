import torch
import torchvision
from torchsummary import summary
import torchvision.transforms as transforms

import numpy as np

from ANP import *
from tqdm import tqdm

import time

#check gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare dataset
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar10_train = torchvision.datasets.CIFAR10('datasets/cifar_10', download=True, transform=transform_train)
cifar10_test = torchvision.datasets.CIFAR10('datasets/cifar_10', train=False, download=True, transform=transform_test)

'''
augment dataset
'''

# we use 0.9 of the whole dataset as the poisoned set
# poisoned_set_ratio = 0.9
# use the very last 1% of train data for the 1% test
poisoned_set_ratio = 0.99

dataset_images = np.array([c[0] for c in cifar10_train])
dataset_labels = np.array([c[1] for c in cifar10_train])

# create a subset of the dataset
l = len(cifar10_train)
indices = np.arange(l)

# don't change this seed (same as the shuffling seed when training backdoored ResNet18)
np.random.seed(594462)
np.random.shuffle(indices)

# since we use 0.9 of the set as backdoored training set, we will use the rest 0.1 as fixing set
keep_indices = indices[int(l * poisoned_set_ratio):]

new_train_set_images = dataset_images[keep_indices, :, :, :]
new_train_set_labels = dataset_labels[keep_indices]

# create a new training set
new_train_set = torch.utils.data.TensorDataset(torch.tensor(new_train_set_images), torch.tensor(new_train_set_labels))

train_loader = torch.utils.data.DataLoader(new_train_set, batch_size=128, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=200, shuffle=False, num_workers=0)

'''
utility functions
'''

# copied from assignment 2
def compute_accuracy(prediction,gt_logits):
    pred_idx = np.argmax(prediction,1,keepdims=True)
    matches = pred_idx == gt_logits[:,None]
    acc = matches.mean()
    return acc

# add a backdoor to a test set to see its efficacy
def introduce_backdoor_test_set(inputs):
    pxl_w = torch.tensor((1.0, 1.0, 1.0))
    pxl_b = torch.tensor((0.0, 0.0, 0.0))
    # pxl_w = (1.0 - 0.4914) / 0.2023
    # pxl_b = (0.0 - 0.4914) / 0.2023
    all_indices = torch.arange(inputs.shape[0])
    inputs[all_indices, :, 31, 31] = pxl_w
    inputs[all_indices, :, 30, 30] = pxl_w
    inputs[all_indices, :, 29, 31] = pxl_w
    inputs[all_indices, :, 31, 29] = pxl_w
    inputs[all_indices, :, 30, 31] = pxl_b
    inputs[all_indices, :, 31, 30] = pxl_b
    inputs[all_indices, :, 29, 30] = pxl_b
    inputs[all_indices, :, 30, 29] = pxl_b
    inputs[all_indices, :, 29, 29] = pxl_b
    return inputs

# pruning threshold
threshold_values = np.linspace(0.05, 0.95, 19)

# function to train an ANP object
def train_anp(anp_system, num_epoches):
    grads_magnitudes = []

    # train for this many epochs
    for epoch in tqdm(range(num_epoches)):
        anp_system.model.train()
    
        i = 0
        total_weight_masks_loss = 0
        for inputs, label in train_loader:
            inputs, label = inputs.to(device), label.to(device)
            # perform perturb step
            weight_masks_loss = anp_system.perturb_step(inputs, label)
            total_weight_masks_loss += weight_masks_loss
            # print(f'epoch: {epoch} | iteration: {i} | weight_mask_loss: {weight_masks_loss}')
            i += 1
    return grads_magnitudes

# then test the network once more
# with the rigidly pruned neurons
def test_backdoor_success(model):
    backdoor_success_ct = 0
    backdoor_item_ct = 0
    with torch.no_grad():
        for inputs, label in test_loader:
            inputs = introduce_backdoor_test_set(inputs).to(device)
            
            pred = model(inputs)
            pred_lbls = np.argmax(pred.cpu().detach().numpy(),1,keepdims=True)

            # be careful to remove the test dataset items that originally were label 0
            # since we don't want to mix them in with testing backdoor on labels 1-9
            backdoor_success_ct += np.sum((pred_lbls == 0).flatten() & (label.numpy() != 0))
            # backdoor_item_ct += inputs.shape[0]
            backdoor_item_ct += np.sum(label.numpy() != 0)
    
    # print(f'Backdoor Success Rate: {backdoor_success_ct/backdoor_item_ct}')
    return backdoor_success_ct/backdoor_item_ct

# then test the model's accuracy on clean data
def test_clean_acc(model):
    total_test_acc = 0
    test_item_ct = 0
    with torch.no_grad():
        for inputs, label in test_loader:
            inputs, label = inputs.to(device), label.to(device)
            
            pred = model(inputs)
            accuracy = compute_accuracy(pred.cpu().detach().numpy(),label.cpu().detach().numpy())
            
            total_test_acc += accuracy * inputs.shape[0]
            test_item_ct += inputs.shape[0]
    # print(f'Test Accuracy: {total_test_acc/test_item_ct}')
    return total_test_acc/test_item_ct

'''
now to run the experiment with 4 different step amounts each with 5 trials
'''
step_amounts = [1, 2, 5, 10]
num_trials = 5
# step_amounts = [1,]
# num_trials = 3

stats = {}

for step_amount in step_amounts:
    for trial_ct in range(num_trials):
        # create and load ResNet18
        num_classes = 10
        res18 = torchvision.models.resnet18().cuda() if torch.cuda.is_available() else torchvision.models.resnet18()
        res18.fc = torch.nn.Linear(res18.fc.in_features, num_classes).cuda() if torch.cuda.is_available() else torch.nn.Linear(res18.fc.in_features, num_classes)
        res18.load_state_dict(torch.load(f'saved_models/ResNet18-CIFAR10-backdoored-5-Epoch-200.pth'))

        # create ANP
        anp_system = ANPWrapper(res18, tradeoff=0.2, lr=0.2, ep=0.4, anp_steps=step_amount)
        
        # train, record time
        start = time.time()
        train_anp(anp_system, 500)
        end = time.time()
        elapsed = end - start
        
        # remove the hooks
        anp_system._remove_hooks()

        # examine efficacy
        asr_list = []
        acc_list = []

        # apply pruning with progressive threshold to see how its ASR and ACC changes
        for threshold_value in threshold_values:
            anp_system._prune_neurons(threshold_value)
            anp_system.model.eval()

            asr = test_backdoor_success(anp_system.model)
            acc = test_clean_acc(anp_system.model)

            # print(f'Threshold value: {threshold_value} | ASR: {asr} | ACC: {acc}')

            asr_list.append(asr)
            acc_list.append(acc)

        if step_amount not in stats:
            stats[step_amount] = []
        stats[step_amount].append({"training_time": elapsed, "ASR": asr_list, "ACC": acc_list})

'''
export to file
'''

import json

stats_json = json.dumps(stats)
savefile = open("experiment_stats_data-1percent_ep-0.4.json", "w")
savefile.write(stats_json)
savefile.close()