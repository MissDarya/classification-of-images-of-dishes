#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 21:47:20 2022

@author: darya
"""



import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import pandas as pd


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}




#data_dir = '/media/darya/Queen/BMSTU/CV/Lab2/SplitFood'
data_dir = '/data/dashat/cv/SplitFood'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#%%
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])



#%%????????????????
def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
    trainLoss = []
    valLoss = []
    
    trainAcc = []
    valAcc = []
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                trainLoss.append(epoch_loss)
                trainAcc.append(epoch_acc.cpu().detach().numpy())
                
            if phase == 'val':
                valLoss.append(epoch_loss)
                valAcc.append(epoch_acc.cpu().detach().numpy())
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, trainLoss, trainAcc, valLoss, valAcc


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()


    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    
                    return
        model.train(mode=was_training)
        
        
def Eval(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    labelList = []
    predList = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labelList.append(labels)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predList.append(preds)
    return labelList, predList

#%%
model_ft = models.resnet18(pretrained=False)
model_ft.load_state_dict(torch.load('/home/dashat/cv/model.pt'))

#model_ft.load_state_dict(torch.load('/media/darya/Queen/BMSTU/CV/Lab2/model.pt'))
#torch.save(model_ft1.state_dict(), '/media/darya/Queen/BMSTU/CV/Lab2/model.pt')


num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 4)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  


model_ft, trainLoss, trainAcc, valLoss, valAcc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)    


#???????????????? ???????????? ????????????
torch.save(model_ft.state_dict(), '/home/dashat/cv/modelBest.pt')  

pathDF = '/home/dashat/cv/df.csv'
df = pd.DataFrame({'trainLoss':trainLoss, 'trainAcc':trainAcc, 
                   'valLoss':valLoss, 'valAcc':valAcc})
df.to_csv(pathDF)

labelList, outputList = Eval(model_ft)

#%%Accuracy
acc = torch.sum(torch.cat(labelList) == torch.cat(outputList))/torch.cat(labelList).size()[0]

print('Accuracy')
print(acc)
#%%F1
f1 = f1_score(torch.cat(labelList).cpu(), torch.cat(outputList).cpu(), average='weighted')
print('F1')
print(f1)

#%%?????????????? ????????????
# cf_matrix = confusion_matrix(torch.cat(labelList), torch.cat(outputList))
# categories = class_names
# sns.heatmap(cf_matrix, annot=True,cmap='Blues',cbar=False,fmt="d",xticklabels = categories,yticklabels=categories)

pathTorch = '/home/dashat/cv/'

#pathTorch = '/media/darya/Queen/BMSTU/CV'

torch.save(torch.cat(labelList),os.path.join(pathTorch, 'label.pt'))
torch.save(torch.cat(outputList),os.path.join(pathTorch, 'outut.pt'))